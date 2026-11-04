import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
DEVICE = os.environ.get("H3_EXPERIMENT_TEST_DEVICE", "cpu")
from comfy.cli_args import args
args.cpu = DEVICE == "cpu"

import torch
import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
import comfy.utils

spec = importlib.util.spec_from_file_location("h3_guider_test", ROOT / "__init__.py")
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
GuiderNode = pack.NODE_CLASS_MAPPINGS["H3ExperimentsMaskedConditioningGuider"]


class Flow(comfy.model_sampling.CONST, comfy.model_sampling.ModelSamplingAV):
    pass


class TestPatcher:
    def __init__(self, model):
        self.model = model
        self.model_options = {}

    def is_dynamic(self):
        return False

    def prepare_state(self, timestep, model_options):
        pass

    def get_free_memory(self, device):
        return 1 << 30

    def apply_hooks(self, hooks):
        return {}

    def get_additional_models_with_key(self, key):
        return []


class TestH3(comfy.model_base.MiniMaxH3):
    """Native H3 conditioning/layout and sampling, with deterministic model output."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.concat_keys = ()
        self.manual_cast_dtype = None
        self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model_sampling = Flow()
        self.diffusion_model = SimpleNamespace(dtype=torch.float32, patch_size=(1, 2, 2), preprocess_text_embeds=lambda x: x)
        self.current_patcher = TestPatcher(self)
        self.calls = []
        self.condition_values = []
        self.square_prediction = False
        self.reference_prediction = False

    def memory_required(self, shape, cond_shapes):
        return 0

    def apply_model(self, x, timestep, c_crossattn, minimax_payload, latent_shapes, **kwargs):
        if x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        self.calls.append((x.clone(), minimax_payload, latent_shapes))
        value = c_crossattn.flatten()[0]
        self.condition_values.append(float(value))
        if self.reference_prediction:
            value = value + minimax_payload["refs"][0]["latent"].mean()
        return torch.ones_like(x) * (value.square() if self.square_prediction else value)


class GuiderTests(unittest.TestCase):
    def setUp(self):
        self.model = TestH3()
        self.video = torch.zeros((1, 24, 7, 2, 4), device=DEVICE)
        self.audio = torch.zeros((1, 32, 2, 37), device=DEVICE)
        self.latent, self.shapes = comfy.utils.pack_latents((self.video, self.audio))

    def cond(self, value, mask=None, **metadata):
        if mask is not None:
            metadata.update(mask=mask, mask_strength=metadata.get("mask_strength", 1.0), set_area_to_bounds=False)
        return [torch.full((1, 2, 4), float(value), device=DEVICE), metadata]

    def run_sample(self, conditions, *, mask_space="frames", blend_mode="predictions", overlap_mix=1.0, denoise_mask=None, sigmas=(1., .5, 0.), callback=None):
        guider = GuiderNode().get_guider(self.model.current_patcher, conditions, mask_space, blend_mode, overlap_mix)[0]
        guider.inner_model = self.model
        guider.conds = {"positive": [c.copy() for c in guider.original_conds["positive"]]}
        original = guider.conds["positive"]
        output = guider.inner_sample(torch.full_like(self.latent, 10), self.latent, torch.device(DEVICE),
                                     comfy.samplers.ksampler("euler"), torch.tensor(sigmas, device=DEVICE),
                                     denoise_mask, callback, True, 123, latent_shapes=self.shapes)
        self.assertIs(guider.conds["positive"], original)
        return comfy.utils.unpack_latents(output, self.shapes)

    def test_original_flat_sampler_mask_failure(self):
        cond = {"mask": torch.ones((1, 8, 16), device=DEVICE)}
        with self.assertRaises(IndexError):
            comfy.samplers.resolve_areas_and_cond_masks_multidim([cond], self.latent.shape[2:], torch.device(DEVICE))

    def test_complementary_regions_and_audio(self):
        mask = torch.tensor([[[0, .25, .75, 1]]], device=DEVICE).expand(1, 2, 4)
        video, audio = self.run_sample([self.cond(2, mask), self.cond(6, 1 - mask)])
        torch.testing.assert_close(video, (6 - 4 * mask).view(1, 1, 1, 2, 4).expand_as(video))
        torch.testing.assert_close(audio, torch.full_like(audio, 4))
        self.assertEqual(len(self.model.calls), 4)
        for a, b in zip(self.model.calls[::2], self.model.calls[1::2]):
            torch.testing.assert_close(a[0], b[0])
            self.assertEqual(a[0].shape[0], 1)

    def test_strengths_normalize_overlap_and_audio(self):
        mask = torch.ones((1, 2, 4), device=DEVICE)
        video, audio = self.run_sample([self.cond(2, mask, mask_strength=3), self.cond(6, mask)])
        torch.testing.assert_close(video, torch.full_like(video, 3))
        torch.testing.assert_close(audio, torch.full_like(audio, 3))

    def test_zero_masks_do_not_remove_audio_or_update_video(self):
        mask = torch.zeros((1, 2, 4), device=DEVICE)
        video, audio = self.run_sample([self.cond(2, mask), self.cond(6, mask)])
        torch.testing.assert_close(video, torch.full_like(video, 10))
        torch.testing.assert_close(audio, torch.full_like(audio, 4))

    def test_unmasked_conditioning_is_global(self):
        video, audio = self.run_sample([self.cond(2), self.cond(6)])
        torch.testing.assert_close(video, torch.full_like(video, 4))
        torch.testing.assert_close(audio, torch.full_like(audio, 4))

    def test_embedding_overlap_precedes_nonlinear_model_prediction(self):
        self.model.square_prediction = True
        mask_a = torch.tensor([[[1., 1., 0., 0.]]], device=DEVICE).expand(1, 2, 4)
        mask_b = torch.tensor([[[0., 1., 1., 0.]]], device=DEVICE).expand(1, 2, 4)
        video, audio = self.run_sample([self.cond(2, mask_a), self.cond(6, mask_b)], blend_mode="embedding_overlap")
        # f(mean(2,6))=16, whereas mean(f(2),f(6))=20.
        expected = torch.tensor([4., 16., 36., 10.], device=DEVICE)
        torch.testing.assert_close(video[0, 0, 0, 0], expected)
        torch.testing.assert_close(audio, torch.full_like(audio, 16))
        self.assertEqual(self.model.condition_values, [2., 6., 4., 2., 6., 4.])

    def test_embedding_overlap_happens_before_h3_text_refinement(self):
        self.model.diffusion_model.preprocess_text_embeds = lambda x: x.square()
        video, audio = self.run_sample([self.cond(2), self.cond(6)], blend_mode="embedding_overlap")
        torch.testing.assert_close(video, torch.full_like(video, 16))
        torch.testing.assert_close(audio, torch.full_like(audio, 16))
        self.assertEqual(self.model.condition_values, [16., 16.])

    def test_overlap_mix_endpoints_and_intermediate_values(self):
        self.model.square_prediction = True
        a = torch.tensor([[[.2, .75, .5, 0.]]], device=DEVICE).expand(1, 2, 4)
        b = torch.tensor([[[0., .25, .5, 0.]]], device=DEVICE).expand(1, 2, 4)
        prediction_blend = torch.tensor([4., 12., 20., 10.], device=DEVICE)
        embedding_overlap = torch.tensor([4., 10., 16., 10.], device=DEVICE)
        for mix in (0., .25, .5, 1.):
            with self.subTest(overlap_mix=mix):
                self.model.condition_values.clear()
                video, audio = self.run_sample([self.cond(2, a), self.cond(6, b)],
                                               blend_mode="embedding_overlap", overlap_mix=mix)
                torch.testing.assert_close(video[0, 0, 0, 0], torch.lerp(prediction_blend, embedding_overlap, mix))
                torch.testing.assert_close(audio, torch.full_like(audio, 20 - 4 * mix))
                if mix == 0:
                    self.assertEqual(self.model.condition_values, [2., 6., 2., 6.])

    def test_predictions_mode_ignores_overlap_mix(self):
        self.model.square_prediction = True
        video, audio = self.run_sample([self.cond(2), self.cond(6)], overlap_mix=.5)
        torch.testing.assert_close(video, torch.full_like(video, 20))
        torch.testing.assert_close(audio, torch.full_like(audio, 20))
        self.assertEqual(self.model.condition_values, [2., 6., 2., 6.])

    def test_average_inside_profile_uses_embedding_branch_and_background(self):
        self.model.square_prediction = True
        mask = torch.tensor([[[1., 1., 0., 0.]]], device=DEVICE).expand(1, 2, 4)
        a, b = pack.NODE_CLASS_MAPPINGS["H3ExperimentsMaskProfile"]().profile(mask, "average_inside")
        video, audio = self.run_sample([self.cond(2, a), self.cond(6, b)], blend_mode="embedding_overlap")
        torch.testing.assert_close(video[0, 0, 0, 0], torch.tensor([16., 16., 36., 36.], device=DEVICE))
        torch.testing.assert_close(audio, torch.full_like(audio, 16))

    def test_five_regions_pairwise_embeddings_and_audio(self):
        self.model.square_prediction = True
        masks = torch.tensor([[1,0,.5,1,0,0,0,1], [0,1,.5,1,0,0,0,0],
                              [0,0,0,1,1,0,0,0], [0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,1,0]], device=DEVICE, dtype=torch.float32).reshape(5,1,2,4)
        conditions = [self.cond(value, mask) for value, mask in zip((1,3,5,7,9), masks)]
        video, audio = self.run_sample(conditions, blend_mode="embedding_overlap")
        expected = torch.tensor([1,9,4,29/3,25,49,81,1],device=DEVICE).reshape(2,4)
        torch.testing.assert_close(video[0,0,0],expected)
        torch.testing.assert_close(audio,torch.full_like(audio,33))
        self.assertEqual(self.model.condition_values,[1.,3.,5.,7.,9.,2.,3.,4.]*2)

    def test_five_region_zero_mix_matches_original_predictions(self):
        self.model.square_prediction = True
        masks = torch.rand(5,7,2,4,device=DEVICE)
        conditions = [self.cond(value, mask) for value,mask in zip((1,3,5,7,9),masks)]
        expected = self.run_sample(conditions,mask_space="latent")
        self.model.condition_values.clear()
        actual = self.run_sample(conditions,mask_space="latent",blend_mode="embedding_overlap",overlap_mix=0)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b)
        self.assertEqual(self.model.condition_values,[1.,3.,5.,7.,9.]*2)

    def test_multi_region_conserves_linear_weights(self):
        generator = torch.Generator(device=DEVICE).manual_seed(42)
        masks = torch.rand(5,7,2,4,generator=generator,device=DEVICE)
        values = torch.tensor([1.,3.,5.,7.,9.],device=DEVICE)
        conditions = [self.cond(value,mask) for value,mask in zip((1,3,5,7,9),masks)]
        video,audio = self.run_sample(conditions,mask_space="latent",blend_mode="embedding_overlap",overlap_mix=.7)
        expected = (masks*values[:,None,None,None]).sum(0)/masks.sum(0)
        torch.testing.assert_close(video[0,0],expected)
        torch.testing.assert_close(audio,torch.full_like(audio,5))

    def test_multi_region_is_order_independent_and_keeps_uncovered_pixels(self):
        self.model.square_prediction = True
        masks = torch.tensor([[.2,.75,.1,0],[.4,.25,.3,0],[.6,0,.6,0]],device=DEVICE).reshape(3,1,1,4).expand(3,1,2,4)
        conditions = [self.cond(value,mask) for value,mask in zip((1,3,5),masks)]
        expected = self.run_sample(conditions,blend_mode="embedding_overlap")
        actual = self.run_sample(conditions[::-1],blend_mode="embedding_overlap")
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b)
        torch.testing.assert_close(actual[0][...,-1],torch.full_like(actual[0][...,-1],10))

    def test_embedding_overlap_preserves_single_branch_strength(self):
        self.model.square_prediction = True
        a = torch.tensor([[[.2, .75, .5, 0.]]], device=DEVICE).expand(1, 2, 4)
        b = torch.tensor([[[0., .25, .5, 1.]]], device=DEVICE).expand(1, 2, 4)
        video, _ = self.run_sample([self.cond(2, a), self.cond(6, b)], blend_mode="embedding_overlap")
        torch.testing.assert_close(video[0, 0, 0, 0], torch.tensor([4., 10., 16., 36.], device=DEVICE))

    def test_embedding_overlap_tracks_moving_masks_on_h3_frame_grid(self):
        self.model.square_prediction = True
        a = torch.arange(22, device=DEVICE, dtype=torch.float32).view(22, 1, 1) / 21
        video, _ = self.run_sample([self.cond(2, a), self.cond(6, 1-a)], blend_mode="embedding_overlap")
        w = torch.tensor([0, 2.5, 6.5, 10.5, 14.5, 17, 19.5], device=DEVICE) / 21
        shared = torch.minimum(w, 1-w)
        expected = (w-shared)*4 + (1-w-shared)*36 + 2*shared*16
        torch.testing.assert_close(video[0, 0, :, 0, 0], expected)

    def test_embedding_overlap_respects_timestep_ranges(self):
        self.model.square_prediction = True
        video, audio = self.run_sample([self.cond(2, start_percent=0, end_percent=.25),
                                       self.cond(6, start_percent=.25, end_percent=1)], blend_mode="embedding_overlap")
        torch.testing.assert_close(video, torch.full_like(video, 36))
        torch.testing.assert_close(audio, torch.full_like(audio, 36))
        self.assertEqual(self.model.condition_values, [2., 6.])

    def test_embedding_overlap_blends_matching_keyframes_without_mutation(self):
        ka = [{"resolved_frame_index":0, "latent":self.video[:,:,:1]+2}]
        kb = [{"resolved_frame_index":0, "latent":self.video[:,:,:1]+6}]
        tags = torch.tensor([0, 1], device=DEVICE)
        a = self.cond(2, minimax_keyframes=ka, minimax_token_tags=tags)
        b = self.cond(6, minimax_keyframes=kb, minimax_token_tags=tags)
        self.run_sample([a,b], blend_mode="embedding_overlap")
        payload = self.model.calls[0][1]
        torch.testing.assert_close(payload["keyframes"][0]["latent"], self.video[:,:,:1]+4)
        torch.testing.assert_close(payload["text_token_tags"], tags)
        self.assertTrue(ka[0]["latent"].eq(2).all())
        self.assertTrue(kb[0]["latent"].eq(6).all())
        self.assertNotIn("mask", a[1])

    def test_embedding_overlap_keeps_denoise_mask_and_preview_contract(self):
        self.model.square_prediction = True
        patcher = comfy.model_patcher.ModelPatcher(self.model, torch.device(DEVICE), torch.device("cpu"))
        self.model.current_patcher = patcher
        guider = GuiderNode().get_guider(patcher, [self.cond(2),self.cond(6)], "frames", "embedding_overlap")[0]
        nested = comfy.nested_tensor.NestedTensor
        previews = []
        result = guider.sample(nested((torch.ones_like(self.video),torch.ones_like(self.audio))),
                               nested((self.video,self.audio)), comfy.samplers.ksampler("euler"),
                               torch.tensor([1.,.5,0.],device=DEVICE),
                               denoise_mask=nested((torch.ones_like(self.video),torch.zeros_like(self.audio))),
                               callback=lambda step,x0,x,count: previews.append((x0,x)),disable_pbar=True,seed=123)
        torch.testing.assert_close(result.tensors[0],torch.full_like(self.video,16))
        torch.testing.assert_close(result.tensors[1],self.audio)
        self.assertEqual(len(previews),2)
        self.assertTrue(all(isinstance(x0,nested) and isinstance(x,nested) for x0,x in previews))

    def test_embedding_overlap_padding_preserves_dtype_and_tags(self):
        average = sys.modules["h3_guider_test.conditioning.overlap_conditioning"].average_conditioning
        a = [torch.full((1,3,5120),2.,device=DEVICE,dtype=torch.bfloat16),{}]
        b = [torch.full((1,2,5120),6.,device=DEVICE,dtype=torch.bfloat16),{}]
        mixed, metadata = average(a,b)
        self.assertEqual(mixed.dtype,torch.bfloat16)
        self.assertEqual(mixed.device,a[0].device)
        torch.testing.assert_close(mixed[:,:2],torch.full_like(mixed[:,:2],4))
        torch.testing.assert_close(mixed[:,2:],torch.full_like(mixed[:,2:],1))
        self.assertEqual(metadata["minimax_token_tags"].tolist(),[1,1,1])

    def test_embedding_overlap_rejects_incompatible_reference_pairing(self):
        average = sys.modules["h3_guider_test.conditioning.overlap_conditioning"].average_conditioning
        with self.assertRaisesRegex(ValueError,"matching image-reference slots"):
            average(self.cond(2,minimax_refs=[{"kind":"image"}]),self.cond(6))
        with self.assertRaisesRegex(ValueError,"image-token positions"):
            average(self.cond(2,minimax_token_tags=torch.tensor([0,1])),self.cond(6))
        with self.assertRaisesRegex(ValueError,"keyframe positions"):
            average(self.cond(2,minimax_keyframes=[{"resolved_frame_index":0}]),
                    self.cond(6,minimax_keyframes=[{"resolved_frame_index":21}]))

    def test_image_reference_overlap_averages_before_inference_without_frame_lock(self):
        self.model.square_prediction = True
        self.model.reference_prediction = True
        mask = torch.tensor([[[1., .5, 0., .5]]], device=DEVICE).expand(1, 2, 4)
        refs_a = [{"kind": "image", "latent_h": 2, "latent_w": 4, "latent": self.video[:, :, :1] + 10}]
        refs_b = [{"kind": "image", "latent_h": 2, "latent_w": 4, "latent": self.video[:, :, :1] + 30}]
        a = self.cond(2, mask, minimax_refs=refs_a)
        b = self.cond(6, 1-mask, minimax_refs=refs_b)
        video, audio = self.run_sample([a, b], blend_mode="embedding_overlap")
        torch.testing.assert_close(video[0, 0, 0, 0], torch.tensor([144., 576., 1296., 576.], device=DEVICE))
        torch.testing.assert_close(audio, torch.full_like(audio, 576))
        for value, call in zip(self.model.condition_values, self.model.calls):
            payload = call[1]
            self.assertNotIn("keyframes", payload)
            self.assertEqual(len(payload["refs"]), 1)
            self.assertEqual(payload["refs"][0]["kind"], "image")
            if value == 4:
                torch.testing.assert_close(payload["refs"][0]["latent"], self.video[:, :, :1] + 20)
                self.assertEqual(len(payload["cond_video_latents"]), 1)
        self.assertTrue(refs_a[0]["latent"].eq(10).all())
        self.assertTrue(refs_b[0]["latent"].eq(30).all())
        mixed_off, audio_off = self.run_sample([a, b], blend_mode="embedding_overlap", overlap_mix=0)
        predicted, audio_predicted = self.run_sample([a, b])
        torch.testing.assert_close(mixed_off, predicted)
        torch.testing.assert_close(audio_off, audio_predicted)

    def test_image_reference_overlap_rejects_unmatched_geometry_and_other_media(self):
        average = sys.modules["h3_guider_test.conditioning.overlap_conditioning"].average_conditioning
        ref = {"kind": "image", "latent_h": 2, "latent_w": 4, "latent": self.video[:, :, :1]}
        a = self.cond(2, minimax_refs=[ref])
        for incompatible in (dict(ref, latent_h=4), dict(ref, latent=self.video[:, :, :2])):
            with self.subTest(incompatible_shape=incompatible["latent"].shape):
                with self.assertRaisesRegex(ValueError, "matching image-reference sizes"):
                    average(a, self.cond(6, minimax_refs=[incompatible]))
        for kind in ("audio", "video", "video_audio"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "image references only"):
                    average(a, self.cond(6, minimax_refs=[dict(ref, kind=kind)]))

    def test_frame_mask_h3_temporal_groups(self):
        mask = torch.arange(22, device=DEVICE, dtype=torch.float32).view(22, 1, 1) / 21
        video, _ = self.run_sample([self.cond(2, mask), self.cond(6, 1 - mask)])
        expected = torch.tensor([0, 2.5, 6.5, 10.5, 14.5, 17, 19.5], device=DEVICE) / 21
        torch.testing.assert_close(video[0, 0, :, 0, 0], 6 - 4 * expected)

    def test_latent_mask_spatial_motion(self):
        mask = torch.zeros((7, 2, 4), device=DEVICE)
        for t in range(7):
            mask[t, :, t % 4] = 1
        video, _ = self.run_sample([self.cond(2, mask), self.cond(6, 1 - mask)], mask_space="latent")
        torch.testing.assert_close(video[0, 0], 6 - 4 * mask)

    def test_real_keyframes_tags_and_layout_preserved(self):
        keyframes = [{"resolved_frame_index": 0, "latent": self.video[:, :, :1]}]
        tags = torch.tensor([[1, 1]], device=DEVICE)
        cond = self.cond(2, torch.ones((1, 2, 4), device=DEVICE), minimax_keyframes=keyframes, minimax_token_tags=tags)
        self.run_sample([cond])
        payload = self.model.calls[0][1]
        self.assertIs(payload["keyframes"], keyframes)
        self.assertIs(payload["text_token_tags"], tags)
        self.assertIsNotNone(payload["layout"])
        self.assertEqual(self.model.calls[0][2], self.shapes)
        self.assertEqual(tuple(cond[1]["mask"].shape), (1, 2, 4))

    def test_denoise_mask_remains_separate_and_protects_audio(self):
        self.latent.fill_(9)
        mask = torch.ones_like(self.video)
        mask[..., :2] = 0
        denoise, _ = comfy.utils.pack_latents((mask, torch.zeros_like(self.audio)))
        video, audio = self.run_sample([self.cond(2)], denoise_mask=denoise)
        torch.testing.assert_close(video[..., :2], torch.full_like(video[..., :2], 9))
        torch.testing.assert_close(video[..., 2:], torch.full_like(video[..., 2:], 2))
        torch.testing.assert_close(audio, torch.full_like(audio, 9))

    def test_timestep_ranges(self):
        video, _ = self.run_sample([self.cond(2, start_percent=0, end_percent=.25),
                                    self.cond(6, start_percent=.25, end_percent=1)])
        torch.testing.assert_close(video, torch.full_like(video, 6))
        self.assertEqual(len(self.model.calls), 2)

    def test_bounds_option_still_uses_full_reference_canvas(self):
        cond = self.cond(2, torch.ones((1, 2, 4), device=DEVICE))
        cond[1]["set_area_to_bounds"] = True
        video, _ = self.run_sample([cond])
        torch.testing.assert_close(video, torch.full_like(video, 2))

    def test_cfg_callback_runs_once_per_step(self):
        seen = []
        self.model.current_patcher.model_options["sampler_post_cfg_function"] = [lambda args: seen.append(args["sigma"]) or args["denoised"]]
        self.run_sample([self.cond(2), self.cond(6)], sigmas=(1., .75, .5, .25, 0.))
        self.assertEqual(len(seen), 4)

    def test_failure_releases_prepared_masks(self):
        conditions = [self.cond(2, torch.ones((1, 2, 4), device=DEVICE))]
        guider = GuiderNode().get_guider(self.model.current_patcher, conditions)[0]
        guider.inner_model = self.model
        guider.conds = {"positive": guider.original_conds["positive"]}
        original = guider.conds["positive"]
        def fail(*args, **kwargs):
            raise RuntimeError("test interruption")
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            guider.inner_sample(self.latent, self.latent, torch.device(DEVICE), SimpleNamespace(sample=fail),
                                torch.tensor([1., 0.], device=DEVICE), None, None, True, 1, latent_shapes=self.shapes)
        self.assertIs(guider.conds["positive"], original)

    def test_full_nested_sampling_entrypoint_and_preview_callback(self):
        patcher = comfy.model_patcher.ModelPatcher(self.model, torch.device(DEVICE), torch.device("cpu"))
        self.model.current_patcher = patcher
        mask = torch.zeros((1, 2, 4), device=DEVICE)
        mask[..., :2] = 1
        guider = GuiderNode().get_guider(patcher, [self.cond(2, mask), self.cond(6, 1 - mask)])[0]
        nested = comfy.nested_tensor.NestedTensor
        previews = []
        result = guider.sample(nested((torch.ones_like(self.video), torch.ones_like(self.audio))),
                               nested((self.video, self.audio)), comfy.samplers.ksampler("euler"),
                               torch.tensor([1., .5, 0.], device=DEVICE),
                               denoise_mask=nested((torch.ones_like(self.video), torch.zeros_like(self.audio))),
                               callback=lambda step, x0, x, count: previews.append((x0, x)), disable_pbar=True, seed=123)
        self.assertIsInstance(result, nested)
        torch.testing.assert_close(result.tensors[0][..., :2], torch.full_like(self.video[..., :2], 2))
        torch.testing.assert_close(result.tensors[0][..., 2:], torch.full_like(self.video[..., 2:], 6))
        torch.testing.assert_close(result.tensors[1], self.audio)
        self.assertEqual(len(previews), 2)
        self.assertTrue(all(isinstance(x0, nested) and isinstance(x, nested) for x0, x in previews))


if __name__ == "__main__":
    torch.set_num_threads(2)
    if DEVICE != "cpu":
        assert torch.cuda.device_count() == 1
        assert "4090" in torch.cuda.get_device_name(0)
        print("Testing on", torch.cuda.get_device_name(0), flush=True)
    unittest.main(verbosity=2)
