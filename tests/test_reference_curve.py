import sys
import unittest

import test_reference_blend as reference_tests
import test_masked_guider as harness
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider

curve = sys.modules["h3_guider_test.reference_curve"]
Blend = reference_tests.Blend


class ReferenceCurveTests(unittest.TestCase):
    def setUp(self):
        fixture = reference_tests.ReferenceBlendTests()
        self.a = fixture.condition(2., 10.)
        self.b = fixture.condition(6., 30.)

    def test_curve_shapes_and_controls(self):
        for mode in ("sine", "triangle"):
            self.assertEqual([curve.curve_value(t, mode) for t in (0., .5, 1.)], [0., 1., 0.])
            self.assertAlmostEqual(curve.curve_value(0., mode, phase=.5), 1.)
            self.assertAlmostEqual(curve.curve_value(.25, mode, cycles=2), 1.)
            self.assertAlmostEqual(curve.curve_value(.5, mode, start=.8, end=.2), .2)
        for mode in ("linear", "ease_in", "ease_out", "ease_in_out"):
            values = [curve.curve_value(i/100, mode) for i in range(101)]
            self.assertEqual(values[0], 0.)
            self.assertEqual(values[-1], 1.)
            self.assertEqual(values, sorted(values))
        self.assertLess(curve.curve_value(.25, "ease_in"), .25)
        self.assertGreater(curve.curve_value(.25, "ease_out"), .25)
        self.assertEqual(curve.curve_value(.5, "ease_in_out"), .5)

    def test_frame_groups_use_actual_reference_time(self):
        weights, frames = curve.temporal_curve_weights(7, "linear")
        self.assertEqual(frames, 22)
        expected = [0., 2.5/21, 6.5/21, 10.5/21, 14.5/21, 17/21, 19.5/21]
        for actual, value in zip(weights, expected): self.assertAlmostEqual(actual, value)
        weights, frames = curve.temporal_curve_weights(37, "sine")
        self.assertEqual(frames, 124)
        self.assertEqual(len(weights), 37)
        self.assertEqual(weights[0], 0.)
        self.assertLess(weights[-1], .005)
        self.assertGreater(max(weights), .99)

    def test_curve_changes_only_reference_time_axis(self):
        mixed, report = Blend().blend(self.a, self.b, .34, .745, "triangle")
        static, _ = Blend().blend(self.a, self.b, .34, .745)
        torch.testing.assert_close(mixed[0][0], static[0][0], rtol=0, atol=0)
        expected = 10 + 20 * torch.tensor([0., 5/21, 13/21, 19/21, 13/21, 8/21, 3/21])
        latent = mixed[0][1]["minimax_refs"][0]["latent"]
        torch.testing.assert_close(latent, expected.view(1,1,7,1,1).expand_as(latent))
        self.assertTrue(self.a[0][1]["minimax_refs"][0]["latent"].eq(10).all())
        self.assertTrue(self.b[0][1]["minimax_refs"][0]["latent"].eq(30).all())
        self.assertIn("22 reference frames -> 7 latent times", report)
        self.assertIn("Context mix stays constant", report)
        self.assertNotIn("reference mix: A=0.255", report)

    def test_constant_mode_keeps_existing_behavior(self):
        old, _ = Blend().blend(self.a, self.b, .34, .745)
        explicit, _ = Blend().blend(self.a, self.b, .34, .745, "constant", .2, .8, 3., -.25)
        torch.testing.assert_close(old[0][0], explicit[0][0], rtol=0, atol=0)
        torch.testing.assert_close(old[0][1]["minimax_refs"][0]["latent"], explicit[0][1]["minimax_refs"][0]["latent"], rtol=0, atol=0)

    def test_identical_inputs_dtype_and_single_image(self):
        self.a[0][1]["minimax_refs"][0]["latent"] = self.a[0][1]["minimax_refs"][0]["latent"].bfloat16()
        mixed, _ = Blend().blend(self.a, self.a, reference_curve="sine", curve_cycles=2.5, curve_phase=-.3)
        actual = mixed[0][1]["minimax_refs"][0]["latent"]
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.eq(10).all())
        weights, frames = curve.temporal_curve_weights(1, "ease_in", start=.25)
        self.assertEqual((weights, frames), ([.25], 1))

    def test_native_sampler_gets_timed_references_with_fixed_context(self):
        model = harness.TestH3()
        model.reference_prediction = True
        conditioning, _ = Blend().blend(self.a, self.b, .5, .745, "linear")
        guider = BasicGuider.execute(model.current_patcher, conditioning)[0]
        guider.inner_model = model
        guider.conds = {"positive": [c.copy() for c in guider.original_conds["positive"]]}
        latent, shapes = harness.comfy.utils.pack_latents((torch.zeros(1,24,7,2,4), torch.zeros(1,32,2,37)))
        guider.inner_sample(torch.ones_like(latent), latent, torch.device("cpu"),
                            harness.comfy.samplers.ksampler("euler"), torch.tensor([1., .5, 0.]),
                            None, None, True, 123, latent_shapes=shapes)
        self.assertEqual(model.condition_values, [4., 4.])
        self.assertEqual(len(model.calls), 2)
        expected = 10 + 20 * torch.tensor([0., 2.5/21, 6.5/21, 10.5/21, 14.5/21, 17/21, 19.5/21])
        for _, payload, _ in model.calls:
            refs = payload["cond_video_latents"]
            self.assertEqual(len(refs), 1)
            torch.testing.assert_close(refs[0], expected.view(1,1,7,1,1).expand_as(refs[0]))
            self.assertNotIn("keyframes", payload)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
