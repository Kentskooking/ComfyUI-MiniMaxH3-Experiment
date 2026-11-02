import unittest

import test_masked_guider as harness
import torch
import torch.nn.functional as F
import comfy.ops
import comfy.supported_models
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from comfy.ldm.modules.attention import attention_pytorch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from h3_guider_test.regional_attention import RegionalRouting, join_conditioning

Node = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalAttentionGuider"]


def tiny_model():
    config = comfy.supported_models.MiniMaxH3(dict(
        hidden_size=32, num_layers=2, token_refiner_num_layers=1, num_attention_heads=1,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=8, timestep_input_dim=8,
        time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=4, dtype=torch.float32))
    config.custom_operations = comfy.ops.manual_cast
    base = comfy.model_base.MiniMaxH3(config, device=torch.device("cpu"))
    for name, parameter in base.diffusion_model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            parameter.fill_(1)
        else:
            parameter.normal_(0, .05)
    base.diffusion_model.rope.inv_freq.copy_(torch.tensor([.1, .03, .01, .003]))
    return base


class RegionalAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        self.contexts = [torch.randn(1, 3, 8), torch.randn(1, 5, 8)]
        self.images = [torch.randn(1, 24, 1, 4, 6), torch.randn(1, 24, 1, 4, 6)]
        self.mask = torch.tensor([[[1., 1., .75, .75, 0., 0.]]]).expand(1, 4, 6)
        self.conditions = [[context, {"mask": mask, "mask_strength": 1., "set_area_to_bounds": False,
                                    "minimax_token_tags": torch.arange(context.shape[1]) % 2,
                                    "minimax_keyframes": [{"resolved_frame_index": 0, "latent": image}]}]
                           for context, mask, image in zip(self.contexts, (self.mask, 1 - self.mask), self.images)]

    def routing(self, conditions=None, video_shape=(1, 24, 2, 4, 6), mask_space="latent", chunk=7):
        conditions = self.conditions if conditions is None else conditions
        merged, lengths, owners = join_conditioning(conditions)
        route = RegionalRouting([c[1] for c in conditions], lengths, owners, video_shape, torch.device("cpu"), mask_space, chunk)
        layout = PackedLayout(sum(lengths), video_shape[2], (video_shape[3] + 1) // 2 * 2,
                              (video_shape[4] + 1) // 2 * 2, 3, keyframes=merged[0][1]["minimax_keyframes"])
        route.bind(layout)
        return route, layout

    def test_join_retains_all_tokens_and_separate_image_guides(self):
        merged, lengths, owners = join_conditioning(self.conditions)
        context, metadata = merged[0]
        self.assertEqual(lengths, (3, 5))
        self.assertEqual(owners, [("cond", 0), ("cond", 1)])
        torch.testing.assert_close(context, torch.cat(self.contexts, dim=1), rtol=0, atol=0)
        self.assertNotIn("mask", metadata)
        self.assertNotIn("mask_strength", metadata)
        self.assertEqual(metadata["minimax_token_tags"].dtype, torch.long)
        for i in range(2):
            self.assertIs(metadata["minimax_keyframes"][i]["latent"], self.images[i])
            self.assertIsNot(metadata["minimax_keyframes"][i], self.conditions[i][1]["minimax_keyframes"][0])

    def test_bias_routes_text_visual_and_anchor_banks_leaving_generated_attention_open(self):
        route, layout = self.routing()
        bias = route.bias(0, layout.seq_len, torch.device("cpu"), torch.float32)
        weights = torch.tensor([[1., 0.], [.75, .25], [0., 1.]]).repeat(4, 1)
        torch.testing.assert_close(route.video_log_weights.exp(), weights)
        for start, stop, bank in route.spans:
            expected = weights[:, bank].log()[:, None].expand(-1, stop - start)
            torch.testing.assert_close(bias[route.video_start:, start:stop], expected)
        torch.testing.assert_close(bias[:, route.audio_start:], torch.zeros_like(bias[:, route.audio_start:]))
        self.assertTrue(torch.isneginf(bias[:3, 3:8]).all())
        torch.testing.assert_close(bias[:3, :3], torch.zeros(3, 3))
        torch.testing.assert_close(bias[route.audio_start:route.audio_stop, :3], torch.full((6, 3), .5).log())

    def test_frame_groups_odd_canvas_and_last_frame_hold(self):
        masks = torch.tensor([1., 0., 1., 0., 1., 0.])[:, None, None].expand(6, 3, 5)
        conditions = [[c[0], {**c[1], "mask": m}] for c, m in zip(self.conditions, (masks, 1 - masks))]
        route, _ = self.routing(conditions, video_shape=(1, 24, 4, 3, 5), mask_space="frames")
        expected = torch.tensor([1., .5, 0., 0.]).repeat_interleave(6)
        torch.testing.assert_close(route.video_log_weights[:, 0].exp(), expected)
        latent, _ = self.routing(conditions, video_shape=(1, 24, 4, 3, 5), mask_space="latent")
        torch.testing.assert_close(latent.video_log_weights[:, 0].exp(), masks[:4, 0, 0].repeat_interleave(6))

    def test_chunked_attention_matches_one_dense_shared_softmax(self):
        route, layout = self.routing()
        q, k, v = (torch.randn(1, 2, layout.seq_len, 16) for _ in range(3))
        preferred = type("Preferred", (), {"function": staticmethod(attention_pytorch)})()
        expected = attention_pytorch(q, k, v, 2, mask=route.bias(0, layout.seq_len, q.device, q.dtype), skip_reshape=True)
        actual = route.attention(None, q, k, v, 2, preferred=preferred,
                                 transformer_options={"minimax_h3_layout": layout}, skip_reshape=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_other_region_reaches_red_through_shared_video_tokens(self):
        route, layout = self.routing()
        bias = route.bias(0, layout.seq_len, torch.device("cpu"), torch.float32)
        q = k = torch.zeros(1, 1, layout.seq_len, 4)
        v = torch.zeros_like(q)
        v[:, :, 3:8] = 10  # Only B's context changes.
        first = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        self.assertEqual(float(first[0, 0, route.video_start, 0]), 0.)  # Pure red cannot directly read B.
        self.assertGreater(float(first[0, 0, route.video_start + 2, 0]), 0.)
        second = F.scaled_dot_product_attention(q, k, first, attn_mask=bias)
        self.assertGreater(float(second[0, 0, route.video_start, 0]), 0.)  # But can interact with B's video region.

    def test_uncovered_regions_and_zero_strength_remain_finite(self):
        conditions = [[c[0], {**c[1], "mask": torch.zeros_like(self.mask), "mask_strength": 0.}] for c in self.conditions]
        route, layout = self.routing(conditions)
        bias = route.bias(0, layout.seq_len, torch.device("cpu"), torch.float32)
        q = torch.randn(1, 1, layout.seq_len, 8)
        self.assertTrue(torch.isfinite(F.scaled_dot_product_attention(q, q, q, attn_mask=bias)).all())

    def test_last_frame_and_audio_guides_keep_their_source_owner(self):
        a = {**self.conditions[0][1], "minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": self.images[0], "audio_latent": torch.ones(1, 32, 2, 2)},
            {"resolved_frame_index": 21, "latent": self.images[0]}]}
        route, layout = self.routing([[self.contexts[0], a], self.conditions[1]])
        self.assertEqual([bank for _, _, bank in route.spans], [0, 1, 0, 0, 0, 1])
        self.assertEqual([kind for _, _, kind in layout.segments],
                         ["text", "cond", "cond_audio", "cond", "cond", "audio", "video"])

    def test_sampling_failure_releases_prepared_routing(self):
        base = tiny_model()
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        guider = Node().get_guider(patcher, self.conditions)[0]
        options = guider.model_options
        guider.inner_model = base
        guider.conds = {"positive": [c.copy() for c in guider.original_conds["positive"]]}
        latent, shapes = comfy.utils.pack_latents((torch.zeros(1, 24, 7, 4, 6), torch.zeros(1, 32, 2, 37)))

        class FailingSampler:
            def sample(self, *args, **kwargs):
                raise RuntimeError("cancelled test")

        with self.assertRaisesRegex(RuntimeError, "cancelled test"):
            guider.inner_sample(latent, latent, torch.device("cpu"), FailingSampler(), torch.tensor([1., 0.]),
                                None, None, True, 123, latent_shapes=shapes)
        self.assertIs(guider.model_options, options)
        self.assertNotIn("h3_regional_attention", options["transformer_options"])

    def test_real_sampler_uses_one_joint_forward_per_step_and_restores_patch_state(self):
        base = tiny_model()
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        original_forward = base.diffusion_model.blocks[0].attn.forward

        def unmasked_patch(*args, **kwargs):
            self.fail("An unmasked attention patch bypassed regional routing")

        patcher.add_object_patch("diffusion_model.blocks.0.attn.forward", unmasked_patch)
        seen = []

        def observe(executor, x, timestep, context, transformer_options, **kwargs):
            payload = kwargs["minimax_payload"]
            seen.append((context.detach().clone(), len(payload["keyframes"])))
            return executor(x, timestep, context, transformer_options, **kwargs)

        patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "test_joint", observe)
        expected = torch.cat([MiniMaxH3Model.preprocess_text_embeds(base.diffusion_model, c) for c in self.contexts], dim=1)
        guider = Node().get_guider(patcher, self.conditions, "frames", 64)[0]
        self.assertIs(patcher.object_patches["diffusion_model.blocks.0.attn.forward"], unmasked_patch)
        self.assertEqual(len(guider.original_conds["positive"]), 1)
        video = torch.zeros(1, 24, 7, 3, 5)
        audio = torch.zeros(1, 32, 2, 37)
        latent = comfy.nested_tensor.NestedTensor([video, audio])
        noise = comfy.nested_tensor.NestedTensor([torch.randn_like(video), torch.randn_like(audio)])
        output = guider.sample(noise, latent, comfy.samplers.ksampler("euler"), torch.tensor([1., .5, 0.]),
                               disable_pbar=True, seed=123)
        self.assertEqual(len(seen), 2)
        for context, guide_count in seen:
            torch.testing.assert_close(context, expected)
            self.assertEqual(guide_count, 2)
        self.assertEqual([z.shape for z in output.unbind()], [video.shape, audio.shape])
        self.assertTrue(all(torch.isfinite(z).all() for z in output.unbind()))
        self.assertNotIn("h3_regional_attention", guider.model_options["transformer_options"])
        self.assertFalse(hasattr(guider, "routing"))
        self.assertEqual(len(self.conditions[0][1]["minimax_keyframes"]), 1)
        guider.model_patcher.unpatch_model(unpatch_weights=False)
        self.assertEqual(base.diffusion_model.blocks[0].attn.forward, original_forward)

    def test_invalid_conditioning_fails_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "two entries"):
            join_conditioning(self.conditions[:1])
        conditions = [[c[0], {**c[1], "minimax_refs": [{"kind": "image"}]}] for c in self.conditions]
        with self.assertRaisesRegex(ValueError, "Omni"):
            join_conditioning(conditions)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
