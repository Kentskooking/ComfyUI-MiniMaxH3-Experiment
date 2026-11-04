import unittest

import test_masked_guider as harness
import test_regional_attention as regional_tests
import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.samplers
from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.modules.attention import attention_pytorch
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from h3_guider_test.sampling.regional_attention import join_conditioning
from h3_guider_test.sampling.regional_interaction import InteractionRouting

Node = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalInteractionGuider"]
OriginalNode = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalAttentionGuider"]


class RegionalInteractionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = regional_tests.RegionalAttentionTests()
        self.fixture.setUp()

    def routing(self, interaction, conditions=None, video_shape=(1, 24, 2, 4, 6), mask_space="latent", chunk=7):
        conditions = self.fixture.conditions if conditions is None else conditions
        joined, lengths, owners = join_conditioning(conditions)
        route = InteractionRouting([c[1] for c in conditions], lengths, owners, video_shape,
                                   torch.device("cpu"), mask_space, chunk, interaction)
        layout = PackedLayout(sum(lengths), video_shape[2], (video_shape[3] + 1) // 2 * 2,
                              (video_shape[4] + 1) // 2 * 2, 3, keyframes=joined[0][1]["minimax_keyframes"])
        route.bind(layout)
        return route, layout

    def test_one_is_exact_original_routing_and_original_schema_is_unchanged(self):
        self.assertNotIn("interaction", OriginalNode.INPUT_TYPES()["required"])
        control = Node.INPUT_TYPES()["required"]["interaction"][1]
        self.assertEqual(control["default"], 1.)
        self.assertEqual(control["display"], "slider")
        for conditions in (self.fixture.conditions, self.fixture.five_conditions()):
            old, layout = self.fixture.routing(conditions)
            new, _ = self.routing(1., conditions)
            self.assertFalse(hasattr(new, "membership"))
            for start in range(0, layout.seq_len, 7):
                stop = min(start + 7, layout.seq_len)
                torch.testing.assert_close(new.bias(start, stop, "cpu", torch.float32),
                                           old.bias(start, stop, "cpu", torch.float32), rtol=0, atol=0)

    def test_moving_five_region_affinity_and_source_readback(self):
        for interaction in (0., .25, .75):
            with self.subTest(interaction=interaction):
                route, layout = self.routing(interaction, self.fixture.five_conditions())
                bias = route.bias(0, layout.seq_len, "cpu", torch.float32)
                weights = route.video_log_weights.exp()
                expected = F.cosine_similarity(weights[:, None, :], weights[None, :, :], dim=-1)
                expected = interaction + (1 - interaction) * expected
                torch.testing.assert_close(bias[route.video_start:, route.video_start:].exp(), expected)
                # First frame starts with source 0; second frame starts with source 4.
                self.assertAlmostEqual(float(bias[route.video_start, route.video_start + 6].exp()), interaction)
                for start, stop, bank in route.spans:
                    membership = F.normalize(weights, dim=-1)[:, bank]
                    expected_readback = interaction + (1 - interaction) * membership
                    torch.testing.assert_close(bias[start:stop, route.video_start:].exp(), expected_readback.expand(stop-start, -1))
                    torch.testing.assert_close(bias[start:stop, route.audio_start:route.audio_stop].exp(),
                                               torch.full((stop-start, route.audio_stop-route.audio_start), interaction))
                torch.testing.assert_close(bias[route.video_start:, route.audio_start:route.audio_stop].exp(),
                                           torch.full((route.video_stop-route.video_start, route.audio_stop-route.audio_start), interaction))
                self.assertTrue((bias[route.audio_start:route.audio_stop, route.audio_start:] == 0).all())

    def test_identical_soft_memberships_stay_open_at_zero(self):
        conditions = [[c[0], {**c[1], "mask": torch.full_like(self.fixture.mask, .5)}] for c in self.fixture.conditions]
        route, layout = self.routing(0., conditions)
        bias = route.bias(0, layout.seq_len, "cpu", torch.float32)
        torch.testing.assert_close(bias[route.video_start:, route.video_start:].exp(), torch.ones(12, 12))

    def test_zero_blocks_disjoint_video_source_and_audio_indirect_paths(self):
        hard = (self.fixture.mask >= .5).float()
        conditions = [[c[0], {**c[1], "mask": m}] for c, m in zip(self.fixture.conditions, (hard, 1 - hard))]
        for interaction in (0., .25, 1.):
            route, layout = self.routing(interaction, conditions)
            bias = route.bias(0, layout.seq_len, "cpu", torch.float32)
            q = k = torch.zeros(1, 1, layout.seq_len, 4)
            values = torch.zeros_like(q)
            values[:, :, 3:8] = 10  # Only B's source changes.
            for _ in range(4):
                values = F.scaled_dot_product_attention(q, k, values, attn_mask=bias)
            if interaction == 0:
                self.assertEqual(float(values[0, 0, route.video_start, 0]), 0.)
                self.assertEqual(float(values[0, 0, 0, 0]), 0.)
            else:
                self.assertGreater(float(values[0, 0, route.video_start, 0]), 0.)

    def test_chunked_attention_matches_dense_for_two_and_five_sources(self):
        for conditions in (self.fixture.conditions, self.fixture.five_conditions()):
            for interaction in (0., .4, 1.):
                route, layout = self.routing(interaction, conditions)
                q, k, v = (torch.randn(1, 2, layout.seq_len, 16) for _ in range(3))
                preferred = type("Preferred", (), {"function": staticmethod(attention_pytorch)})()
                expected = attention_pytorch(q, k, v, 2, mask=route.bias(0, layout.seq_len, "cpu", q.dtype), skip_reshape=True)
                actual = route.attention(None, q, k, v, 2, preferred=preferred,
                                         transformer_options={"minimax_h3_layout": layout}, skip_reshape=True)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_frame_conversion_odd_canvas_and_uncovered_regions(self):
        masks = torch.tensor([1., 0., 1., 0., 1., 0.])[:, None, None].expand(6, 3, 5)
        conditions = [[c[0], {**c[1], "mask": m}] for c, m in zip(self.fixture.conditions, (masks, 1 - masks))]
        old, _ = self.fixture.routing(conditions, (1, 24, 4, 3, 5), "frames")
        new, _ = self.routing(.5, conditions, (1, 24, 4, 3, 5), "frames")
        torch.testing.assert_close(new.video_log_weights, old.video_log_weights, rtol=0, atol=0)
        for strength in (0., 1.):
            sources = [[c[0], {**c[1], "mask": torch.zeros_like(self.fixture.mask), "mask_strength": strength}]
                       for c in self.fixture.conditions]
            route, layout = self.routing(0., sources)
            bias = route.bias(0, layout.seq_len, "cpu", torch.float32)
            q = torch.randn(1, 1, layout.seq_len, 8)
            self.assertTrue(torch.isfinite(F.scaled_dot_product_attention(q, q, q, attn_mask=bias)).all())
            self.assertTrue((bias[route.video_start:, route.video_start:] == 0).all())

    def sample(self, base, conditions, node, interaction=None):
        device = torch.device(harness.DEVICE)
        patcher = ModelPatcher(base, device, torch.device("cpu"))
        original_forward = base.diffusion_model.blocks[0].attn.forward
        calls = []

        def observe(executor, x, timestep, context, transformer_options, **kwargs):
            calls.append((context.shape[1], len(kwargs["minimax_payload"]["keyframes"])))
            return executor(x, timestep, context, transformer_options, **kwargs)

        patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "test_interaction", observe)
        kwargs = {} if interaction is None else {"interaction": interaction}
        guider = node().get_guider(patcher, conditions, "frames", 64, **kwargs)[0]
        self.assertEqual(patcher.object_patches, {})
        video, audio = torch.zeros(1, 24, 7, 3, 5), torch.zeros(1, 32, 2, 37)
        torch.manual_seed(123)
        latent = comfy.nested_tensor.NestedTensor([video, audio])
        noise = comfy.nested_tensor.NestedTensor([torch.randn_like(video), torch.randn_like(audio)])
        output = guider.sample(noise, latent, comfy.samplers.ksampler("euler"), torch.tensor([1., .5, 0.]),
                               disable_pbar=True, seed=123)
        self.assertEqual(calls, [(sum(c[0].shape[1] for c in conditions), len(conditions))] * 2)
        self.assertEqual([z.shape for z in output.unbind()], [video.shape, audio.shape])
        self.assertTrue(all(torch.isfinite(z).all() for z in output.unbind()))
        self.assertNotIn("h3_regional_attention", guider.model_options["transformer_options"])
        self.assertFalse(hasattr(guider, "routing"))
        guider.model_patcher.unpatch_model(unpatch_weights=False)
        self.assertEqual(base.diffusion_model.blocks[0].attn.forward, original_forward)
        return output

    def test_native_sampler_one_matches_original_and_slider_changes_output(self):
        base = regional_tests.tiny_model()
        original = self.sample(base, self.fixture.conditions, OriginalNode)
        current = self.sample(base, self.fixture.conditions, Node, 1.)
        reduced = self.sample(base, self.fixture.conditions, Node, .25)
        restored = self.sample(base, self.fixture.conditions, OriginalNode)
        for a, b, d in zip(original.unbind(), current.unbind(), restored.unbind()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, d, rtol=0, atol=0)
        self.assertFalse(torch.equal(current.unbind()[0], reduced.unbind()[0]))

    def test_five_source_native_sampler_zero_and_half_still_use_one_forward_per_step(self):
        base = regional_tests.tiny_model()
        conditions = self.fixture.five_conditions()
        self.sample(base, conditions, Node, 0.)
        self.sample(base, conditions, Node, .5)

    def test_failure_does_not_leave_new_routing_on_old_node_or_model(self):
        base = regional_tests.tiny_model()
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        guider = Node().get_guider(patcher, self.fixture.conditions, interaction=.5)[0]
        options = guider.model_options
        guider.inner_model = base
        guider.conds = {"positive": [c.copy() for c in guider.original_conds["positive"]]}
        latent, shapes = comfy.utils.pack_latents((torch.zeros(1, 24, 7, 4, 6), torch.zeros(1, 32, 2, 37)))

        class FailingSampler:
            def sample(self, *args, **kwargs):
                raise RuntimeError("cancelled interaction test")

        with self.assertRaisesRegex(RuntimeError, "cancelled interaction test"):
            guider.inner_sample(latent, latent, torch.device("cpu"), FailingSampler(), torch.tensor([1., 0.]),
                                None, None, True, 123, latent_shapes=shapes)
        self.assertIs(guider.model_options, options)
        self.assertNotIn("h3_regional_attention", options["transformer_options"])
        self.assertEqual(patcher.object_patches, {})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
