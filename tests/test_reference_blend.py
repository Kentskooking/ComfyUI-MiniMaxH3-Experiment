import copy
import unittest

import test_masked_guider as harness

import torch
import comfy.samplers
from comfy_extras.nodes_custom_sampler import BasicGuider
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, _empty_av_latent

Blend = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsOmniReferenceBlend"]


class ReferenceBlendTests(unittest.TestCase):
    def condition(self, context, video, length=4, dtype=torch.float32):
        return [[torch.full((1, length, 8), context, dtype=dtype), {
            "minimax_token_tags": torch.tensor([0, 0] + [1] * (length-2)),
            "minimax_refs": [{"kind": "video", "latent_t": 7, "latent_h": 2, "latent_w": 4,
                              "latent": torch.full((1, 24, 7, 2, 4), video, dtype=dtype),
                              "ref_audio_t": 0, "audio_latent": None}],
        }]]

    def test_independent_mix_controls_and_endpoint_values(self):
        a, b = self.condition(2., 10.), self.condition(6., 30.)
        for context_mix, reference_mix in ((0., 0.), (1., 1.), (.25, .75), (.5, 0.), (0., .5)):
            with self.subTest(context_mix=context_mix, reference_mix=reference_mix):
                mixed, report = Blend().blend(a, b, context_mix, reference_mix)
                torch.testing.assert_close(mixed[0][0], torch.full_like(a[0][0], 2 + context_mix * 4))
                torch.testing.assert_close(mixed[0][1]["minimax_refs"][0]["latent"],
                                           torch.full_like(a[0][1]["minimax_refs"][0]["latent"], 10 + reference_mix * 20))
                self.assertIn("Context RMS distance: A-B=4", report)
                self.assertIn("B tag mismatches=0", report)

    def test_b_truncation_uses_a_length_and_tags(self):
        a, b = self.condition(2., 10., 4), self.condition(6., 30., 7)
        b[0][0][:, 4:] = 99
        mixed, report = Blend().blend(a, b, 1., 1.)
        self.assertEqual(mixed[0][0].shape, (1, 4, 8))
        self.assertTrue(mixed[0][0].eq(6).all())
        torch.testing.assert_close(mixed[0][1]["minimax_token_tags"], a[0][1]["minimax_token_tags"])
        self.assertIn("B truncated=3, zero-padded=0", report)

    def test_short_b_is_zero_padded(self):
        mixed, report = Blend().blend(self.condition(2., 10., 6), self.condition(6., 30., 4), .5, .5)
        self.assertTrue(mixed[0][0][:, :4].eq(4).all())
        self.assertTrue(mixed[0][0][:, 4:].eq(1).all())
        self.assertIn("B truncated=0, zero-padded=2", report)

    def test_tag_mismatch_is_reported_without_alignment(self):
        a, b = self.condition(2., 10.), self.condition(6., 30.)
        b[0][1]["minimax_token_tags"] = torch.tensor([1, 1, 0, 0])
        mixed, report = Blend().blend(a, b)
        torch.testing.assert_close(mixed[0][1]["minimax_token_tags"], a[0][1]["minimax_token_tags"])
        self.assertIn("B tag mismatches=4", report)

    def test_identical_inputs_and_no_mutation(self):
        a = self.condition(2., 10., dtype=torch.bfloat16)
        before = copy.deepcopy(a)
        a[0][1]["mask"] = torch.ones(1, 8, 8)
        for mix in (0., .13, .5, 1.):
            mixed, report = Blend().blend(a, a, mix, 1-mix)
            torch.testing.assert_close(mixed[0][0], before[0][0], rtol=0, atol=0)
            torch.testing.assert_close(mixed[0][1]["minimax_refs"][0]["latent"], before[0][1]["minimax_refs"][0]["latent"], rtol=0, atol=0)
            self.assertEqual(mixed[0][0].dtype, torch.bfloat16)
            self.assertNotIn("mask", mixed[0][1])
            self.assertIn("output-A=0, output-B=0", report)
        self.assertIn("mask", a[0][1])
        torch.testing.assert_close(a[0][0], before[0][0])
        torch.testing.assert_close(a[0][1]["minimax_refs"][0]["latent"], before[0][1]["minimax_refs"][0]["latent"])

    def test_invalid_connections_have_actionable_errors(self):
        a, b = self.condition(2., 10.), self.condition(6., 30.)
        with self.assertRaisesRegex(ValueError, "one conditioning entry"):
            Blend().blend(a + b, b)
        b[0][1]["minimax_refs"] = []
        with self.assertRaisesRegex(ValueError, "video VAE connected"):
            Blend().blend(a, b)
        b = self.condition(6., 30.)
        b[0][1]["minimax_refs"][0]["latent"] = torch.zeros(1, 24, 12, 2, 4)
        with self.assertRaisesRegex(ValueError, "same frame count"):
            Blend().blend(a, b)
        b = self.condition(6., 30.)
        b[0][1]["minimax_refs"][0]["kind"] = "video_audio"
        with self.assertRaisesRegex(ValueError, "Disconnect reference audio"):
            Blend().blend(a, b)

    def test_native_basic_guider_receives_mixture_before_refinement(self):
        model = harness.TestH3()
        refined_inputs = []
        def refine(x):
            refined_inputs.append(x.clone())
            return x.square()
        model.diffusion_model.preprocess_text_embeds = refine
        model.reference_prediction = True
        model.square_prediction = True
        a, b = self.condition(2., 10.), self.condition(6., 30.)
        a[0][0][:, 2:] = 5
        b[0][0][:, 2:] = 17
        conditioning, _ = Blend().blend(a, b, .25, .75)
        guider = BasicGuider.execute(model.current_patcher, conditioning)[0]
        guider.inner_model = model
        guider.conds = {"positive": [c.copy() for c in guider.original_conds["positive"]]}
        video, audio = torch.zeros(1, 24, 7, 2, 4), torch.zeros(1, 32, 2, 37)
        latent, shapes = harness.comfy.utils.pack_latents((video, audio))
        result = guider.inner_sample(torch.ones_like(latent), latent, torch.device("cpu"),
                                     comfy.samplers.ksampler("euler"), torch.tensor([1., .5, 0.]),
                                     None, None, True, 123, latent_shapes=shapes)
        # Qwen mixture 3 is refined to 9; reference mixture 25 reaches the same
        # model call. f(9 + 25) = 1156, not a blend of separate A/B predictions.
        output_video, output_audio = harness.comfy.utils.unpack_latents(result, shapes)
        torch.testing.assert_close(output_video, torch.full_like(video, 1156.))
        torch.testing.assert_close(output_audio, torch.full_like(audio, 1156.))
        self.assertEqual(model.condition_values, [9., 9.])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(refined_inputs), 1)
        self.assertTrue(refined_inputs[0][:, :2].eq(3).all())
        self.assertTrue(refined_inputs[0][:, 2:].eq(8).all())
        for _, payload, _ in model.calls:
            self.assertNotIn("keyframes", payload)
            self.assertEqual(len(payload["refs"]), 1)
            self.assertEqual(payload["refs"][0]["kind"], "video")
            self.assertTrue(payload["cond_video_latents"][0].eq(25).all())
            self.assertEqual(len(payload["cond_audio_latents"]), 0)
            self.assertIsNotNone(payload["layout"])

    def test_native_video_reference_encoder_payload(self):
        class Clip:
            def tokenize(self, prompt, minimax_ref_items):
                self.items = minimax_ref_items
                return float(prompt) + float(minimax_ref_items[0]["data"].mean())

            def encode_from_tokens_scheduled(self, value):
                return [[torch.full((1, 4, 8), value), {"minimax_token_tags": torch.tensor([0, 0, 1, 1])}]]

        class Vae:
            def encode(self, frames):
                latent, _ = _empty_av_latent(frames.shape[2], frames.shape[1], frames.shape[0])
                return torch.ones_like(latent["samples"].tensors[0]) * frames.mean()

        clip_a, clip_b = Clip(), Clip()
        a = MiniMaxH3ReferenceToVideo.execute(clip_a, "2", 64, 32, 22, vae=Vae(),
                                             ref_videos={"ref_video_0": torch.zeros(22, 32, 64, 3)})
        b = MiniMaxH3ReferenceToVideo.execute(clip_b, "6", 64, 32, 22, vae=Vae(),
                                             ref_videos={"ref_video_0": torch.ones(22, 32, 64, 3)})
        mixed, _ = Blend().blend(a[0], b[0])
        torch.testing.assert_close(mixed[0][0], torch.full_like(mixed[0][0], 4.5))
        torch.testing.assert_close(mixed[0][1]["minimax_refs"][0]["latent"], torch.full_like(mixed[0][1]["minimax_refs"][0]["latent"], .5))
        self.assertEqual(clip_a.items[0]["data"].shape[0], 2)
        self.assertEqual(clip_b.items[0]["timestamps"], [0., .5])
        self.assertTrue(a[1]["samples"].tensors[0].eq(0).all())
        self.assertNotIn("minimax_keyframes", mixed[0][1])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
