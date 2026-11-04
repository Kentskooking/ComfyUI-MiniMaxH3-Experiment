import importlib.util
from pathlib import Path
import unittest

import torch
from torchvision.transforms.v2 import functional as TF

spec = importlib.util.spec_from_file_location("h3_mask_profile_test", Path(__file__).resolve().parents[1] / "masks" / "mask_profile.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Profile = module.H3MaskProfile


class MaskProfileTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(3, 64, 160)
        self.mask[:, :, 30:130] = 1

    def test_all_modes_complement_and_preserve_input(self):
        before = self.mask.clone()
        for mode in ("original", "gaussian", "interior_gradient", "average_inside"):
            with self.subTest(mode=mode):
                a, b = Profile().profile(self.mask, mode, transition_width=30, outer_fade=10)
                self.assertEqual(a.shape, self.mask.shape)
                self.assertEqual(a.dtype, torch.float32)
                self.assertEqual(a.device, self.mask.device)
                self.assertTrue(bool(((a >= 0) & (a <= 1)).all()))
                torch.testing.assert_close(a+b, torch.ones_like(a))
                torch.testing.assert_close(self.mask, before)

    def test_original_preserves_soft_weights(self):
        mask = torch.tensor([[0., .25, .5, 1.]], dtype=torch.float64)
        a, b = Profile().profile(mask, "original")
        torch.testing.assert_close(a, mask.float().unsqueeze(0))
        torch.testing.assert_close(b, 1-mask.float().unsqueeze(0))

    def test_average_inside_has_no_pure_a(self):
        a, b = Profile().profile(self.mask, "average_inside")
        torch.testing.assert_close(a, self.mask*.5)
        self.assertEqual(float(a.max()), .5)
        self.assertEqual(float(b.min()), .5)

    def test_interior_core_boundary_and_outer_tail(self):
        a, _ = Profile().profile(self.mask, transition_width=30, outer_fade=10)
        row = a[0, 32]
        self.assertEqual(float(row[80]), 1.)
        self.assertEqual(float(row[19]), 0.)
        self.assertLess(float(row[29]), .25)
        self.assertGreater(float(row[30]), .25)
        self.assertLess(float(row[30]), .26)
        self.assertTrue(bool((row[1:81] >= row[:80]).all()))
        torch.testing.assert_close(row, row.flip(0))

    def test_no_outer_fade_reaches_zero_at_boundary(self):
        a, _ = Profile().profile(self.mask, transition_width=30, outer_fade=0, curve="linear")
        self.assertEqual(float(a[0, 32, 29]), 0.)
        self.assertAlmostEqual(float(a[0, 32, 30]), .5/30, places=6)
        self.assertEqual(float(a[0, 32, 80]), 1.)

    def test_no_fades_is_binary_mask(self):
        a, _ = Profile().profile(self.mask, transition_width=0, outer_fade=0)
        torch.testing.assert_close(a, self.mask)

    def test_empty_full_and_frame_edges(self):
        mask = self.mask.clone()
        mask[0] = 0
        mask[1] = 1
        mask[2] = 0
        mask[2, :, :80] = 1
        a, _ = Profile().profile(mask, transition_width=30, outer_fade=10)
        self.assertTrue(bool(a[0].eq(0).all()))
        self.assertTrue(bool(a[1].eq(1).all()))
        self.assertTrue(bool(a[2, :, 0].eq(1).all()))

    def test_moving_mask_keeps_spatial_profile(self):
        self.mask[1] = self.mask[0].roll(10, dims=1)
        self.mask[2] = 0
        a, _ = Profile().profile(self.mask, transition_width=30, outer_fade=10)
        torch.testing.assert_close(a[1], a[0].roll(10, dims=1))
        self.assertTrue(bool(a[2].eq(0).all()))

    def test_gaussian_matches_essentials_kernel(self):
        for blur in (0, 2, 31):
            with self.subTest(blur=blur):
                a, _ = Profile().profile(self.mask, "gaussian", blur=blur)
                kernel = blur + 1 if blur % 2 == 0 else blur
                expected = TF.gaussian_blur(self.mask.unsqueeze(1), kernel).squeeze(1) if blur else self.mask
                torch.testing.assert_close(a, expected, atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
