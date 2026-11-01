import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
DEVICE = os.environ.get("H3_EXPERIMENT_TEST_DEVICE", "cpu")
from comfy.cli_args import args
args.cpu = DEVICE == "cpu"

import torch
import comfy.nested_tensor
import comfy.utils
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

spec = importlib.util.spec_from_file_location("h3_experiments_test", ROOT / "__init__.py")
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
Composite = pack.NODE_CLASS_MAPPINGS["H3ExperimentsLatentCompositeMasked"]
NestedTensor = comfy.nested_tensor.NestedTensor


class CompositeTests(unittest.TestCase):
    def latent(self, value, *, joint=True, dtype=torch.float32, shape=(1, 24, 7, 2, 4)):
        video = torch.full(shape, value, device=DEVICE, dtype=dtype)
        audio = torch.full((1, 32, 2, 37), value + 10, device=DEVICE, dtype=dtype)
        return {"samples": NestedTensor((video, audio)) if joint else video, "marker": "keep"}

    def composite(self, dest, source, mask=None, **kwargs):
        return Composite().composite(dest, source, mask=mask, **kwargs)[0]

    def video(self, latent):
        samples = latent["samples"]
        return samples.tensors[0] if isinstance(samples, NestedTensor) else samples

    def noise_mask(self, latent, value, audio_value=0):
        video = self.video(latent)
        mask = torch.full((1, 1, *video.shape[2:]), value, dtype=torch.float32, device=DEVICE)
        if isinstance(latent["samples"], NestedTensor):
            audio = torch.full((1, 1, 2, 37), audio_value, dtype=torch.float32, device=DEVICE)
            latent["noise_mask"] = NestedTensor((mask, audio))
        else:
            latent["noise_mask"] = mask
        return latent

    def test_static_white_black_and_soft_masks(self):
        dest, source = self.latent(2), self.latent(6)
        for value in (0.0, 0.25, 1.0):
            with self.subTest(value=value):
                result = self.composite(dest, source, torch.full((1, 8, 16), value, device=DEVICE))
                torch.testing.assert_close(self.video(result), torch.full_like(self.video(dest), 2 + 4 * value))

    def test_h3_pixel_frame_groups_keep_time_distinct(self):
        mask = torch.arange(22, dtype=torch.float32, device=DEVICE).view(22, 1, 1) / 21
        result = self.composite(self.latent(0), self.latent(1), mask)
        expected = torch.tensor([0, 2.5, 6.5, 10.5, 14.5, 17, 19.5], device=DEVICE) / 21
        torch.testing.assert_close(self.video(result)[0, 0, :, 0, 0], expected)

    def test_moving_spatial_mask_tracks_both_axes(self):
        mask = torch.zeros((7, 2, 4), device=DEVICE)
        mask[0, 0, :2] = 1
        mask[-1, 1, 2:] = 1
        result = self.composite(self.latent(0), self.latent(1), mask, mask_space="latent")
        torch.testing.assert_close(self.video(result)[0, 0], mask)

    def test_short_sequence_holds_last_mask(self):
        mask = torch.tensor([0.0, 1.0], device=DEVICE).view(2, 1, 1)
        result = self.composite(self.latent(0), self.latent(1), mask)
        torch.testing.assert_close(self.video(result)[0, 0, :, 0, 0], torch.tensor([0., 1, 1, 1, 1, 1, 1], device=DEVICE))

    def test_excess_mask_frames_do_not_retime_the_clip(self):
        mask = torch.cat((torch.zeros(22, 1, 1), torch.ones(17, 1, 1))).to(DEVICE)
        result = self.composite(self.latent(0), self.latent(1), mask)
        self.assertEqual(self.video(result).count_nonzero().item(), 0)

    def test_destination_audio_and_metadata_are_retained(self):
        dest, source = self.latent(0), self.latent(1)
        result = self.composite(dest, source)
        self.assertIs(result["samples"].tensors[1], dest["samples"].tensors[1])
        self.assertEqual(result["marker"], "keep")
        self.assertIsNot(result, dest)
        self.assertEqual(self.video(dest).count_nonzero().item(), 0)
        self.assertTrue(self.video(source).eq(1).all())
        self.assertNotIn("noise_mask", result)

    def test_joint_destination_accepts_video_only_source(self):
        dest = self.latent(0)
        result = self.composite(dest, self.latent(1, joint=False))
        self.assertTrue(self.video(result).eq(1).all())
        self.assertIs(result["samples"].tensors[1], dest["samples"].tensors[1])

    def test_video_only_destination_stays_video_only(self):
        result = self.composite(self.latent(0, joint=False), self.latent(1))
        self.assertIsInstance(result["samples"], torch.Tensor)
        self.assertEqual(tuple(result["samples"].shape), (1, 24, 7, 2, 4))

    def test_identical_sources_are_unchanged(self):
        source = self.latent(0.75)
        mask = torch.rand((22, 8, 16), device=DEVICE)
        result = self.composite(source, source, mask)
        torch.testing.assert_close(self.video(result), self.video(source), rtol=0, atol=0)

    def test_swapping_sources_and_inverting_mask_is_equivalent(self):
        mask = torch.rand((22, 8, 16), device=DEVICE)
        a, b = self.latent(-2), self.latent(3)
        left = self.composite(a, b, mask)
        right = self.composite(b, a, 1 - mask)
        torch.testing.assert_close(self.video(left), self.video(right))

    def test_complementary_denoise_masks_follow_selected_content(self):
        dest, source = self.noise_mask(self.latent(0), 0), self.noise_mask(self.latent(1), 0)
        alpha = torch.zeros((7, 2, 4), device=DEVICE)
        alpha[:, :, 2:] = 1
        dest["noise_mask"].tensors[0][:] = 1 - alpha
        source["noise_mask"].tensors[0][:] = alpha
        result = self.composite(dest, source, alpha, mask_space="latent")
        self.assertTrue(result["noise_mask"].tensors[0].eq(1).all())
        self.assertIs(result["noise_mask"].tensors[1], dest["noise_mask"].tensors[1])

    def test_missing_denoise_mask_means_generate(self):
        dest = self.noise_mask(self.latent(0), 0, audio_value=0)
        result = self.composite(dest, self.latent(1), torch.full((1, 2, 4), .25, device=DEVICE))
        self.assertTrue(result["noise_mask"].tensors[0].eq(.25).all())
        self.assertTrue(result["noise_mask"].tensors[1].eq(0).all())

    def test_source_mask_does_not_replace_destination_audio_mask(self):
        source = self.noise_mask(self.latent(1), 0, audio_value=0)
        result = self.composite(self.latent(0), source)
        self.assertTrue(result["noise_mask"].tensors[0].eq(0).all())
        self.assertTrue(result["noise_mask"].tensors[1].eq(1).all())

    def test_plain_destination_noise_mask_gets_audio_generate_default(self):
        dest = self.latent(0)
        dest["noise_mask"] = torch.zeros((7, 2, 4), device=DEVICE)
        result = self.composite(dest, self.latent(1), torch.zeros((1, 2, 4), device=DEVICE))
        self.assertTrue(result["noise_mask"].tensors[0].eq(0).all())
        self.assertTrue(result["noise_mask"].tensors[1].eq(1).all())

    def test_keep_and_remove_noise_mask_modes(self):
        dest = self.noise_mask(self.latent(0), .3)
        source = self.noise_mask(self.latent(1), .7)
        kept = self.composite(dest, source, noise_mask_mode="destination")
        self.assertIs(kept["noise_mask"], dest["noise_mask"])
        cleared = self.composite(dest, source, noise_mask_mode="none")
        self.assertNotIn("noise_mask", cleared)
        self.assertIn("noise_mask", dest)

    def test_dtype_and_fractional_mask_precision(self):
        dest = self.noise_mask(self.latent(0, dtype=torch.bfloat16), .9995)
        source = self.noise_mask(self.latent(1, dtype=torch.float32), .9995)
        mask = torch.full((22, 2, 4), .5, device=DEVICE)
        result = self.composite(dest, source, mask)
        self.assertEqual(self.video(result).dtype, torch.bfloat16)
        self.assertEqual(result["noise_mask"].tensors[0].dtype, torch.float32)
        self.assertTrue(result["noise_mask"].tensors[0].eq(.9995).all())
        self.assertTrue(mask.eq(.5).all())

    def test_singleton_source_broadcasts(self):
        result = self.composite(self.latent(0), self.latent(1, shape=(1, 24, 1, 1, 1)))
        self.assertEqual(tuple(self.video(result).shape), (1, 24, 7, 2, 4))
        self.assertTrue(self.video(result).eq(1).all())

    def test_wrong_family_and_mismatched_canvases_report_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "H3 video latents"):
            self.composite(self.latent(0), self.latent(1, shape=(1, 16, 7, 2, 4)))
        with self.assertRaisesRegex(ValueError, "canvas and frame count"):
            self.composite(self.latent(0), self.latent(1, shape=(1, 24, 8, 2, 4)))

    def test_native_h3_canvas_survives_sampler_pack_unpack(self):
        dest, frame_count = _empty_av_latent(64, 32, 124)
        source, _ = _empty_av_latent(64, 32, 124)
        dest["samples"] = dest["samples"].to(DEVICE)
        source["samples"] = source["samples"].to(DEVICE)
        source["samples"].tensors[0].fill_(1)
        result = self.composite(dest, source, torch.full((124, 32, 64), .5, device=DEVICE))
        packed, shapes = comfy.utils.pack_latents(result["samples"].unbind())
        unpacked = comfy.utils.unpack_latents(packed, shapes)
        self.assertEqual(frame_count, 124)
        self.assertEqual(tuple(unpacked[0].shape), (1, 24, 37, 2, 4))
        self.assertEqual(tuple(unpacked[1].shape), (1, 32, 2, 207))
        self.assertTrue(unpacked[0].eq(.5).all())
        self.assertTrue(unpacked[1].eq(0).all())


if __name__ == "__main__":
    torch.set_num_threads(2)
    if DEVICE != "cpu":
        assert torch.cuda.device_count() == 1
        assert "4090" in torch.cuda.get_device_name(0)
        print("Testing on", torch.cuda.get_device_name(0), flush=True)
    unittest.main(verbosity=2)
