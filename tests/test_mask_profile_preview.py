import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from comfy.cli_args import args
args.cpu = True

import numpy as np
from PIL import Image
import torch

spec = importlib.util.spec_from_file_location("h3_preview_test", ROOT / "__init__.py")
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
Profile = pack.H3MaskProfile
Preview = pack.H3MaskProfilePreview


class MaskProfilePreviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.node = Preview()
        self.node.output_dir = self.directory.name
        self.mask = torch.zeros(5, 40, 100)
        for i in range(5):
            self.mask[i, :, 10 + i * 8:50 + i * 8] = 1

    def test_selected_frame_matches_full_batch_in_all_modes(self):
        before = self.mask.clone()
        for mode in ("original", "gaussian", "interior_gradient", "average_inside"):
            with self.subTest(mode=mode):
                a, b = Profile().profile(self.mask, mode, 16, 8, "smoothstep", 32)
                with patch.object(Profile, "profile", wraps=Profile().profile) as profile:
                    result = self.node.preview(self.mask, 2, mode, 16, 8, "smoothstep", 32)
                self.assertEqual(profile.call_args.args[0].shape, (1, 40, 100))
                panel = result["result"][0][0, 32:72, :, 0]
                expected = torch.cat((self.mask[2], a[2], b[2]), dim=1)
                torch.testing.assert_close(panel, expected, atol=.5/255 + 1e-7, rtol=0)
                descriptor = result["ui"]["images"][0]
                self.assertEqual(descriptor["type"], "temp")
                with Image.open(Path(self.directory.name) / descriptor["filename"]) as saved:
                    np.testing.assert_array_equal(np.asarray(saved), (result["result"][0][0].numpy() * 255).astype(np.uint8))
        torch.testing.assert_close(self.mask, before)

    def test_out_of_range_index_holds_last_frame(self):
        result = self.node.preview(self.mask, 999, "original")
        panel = result["result"][0][0, 32:72, :100, 0]
        torch.testing.assert_close(panel, self.mask[-1])

    def test_static_mask_and_empty_batch(self):
        result = self.node.preview(self.mask[1], 0, "average_inside")
        panel = result["result"][0][0, 32:72, 100:200, 0]
        torch.testing.assert_close(panel, self.mask[1] * .5, atol=.5/255 + 1e-7, rtol=0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.node.preview(self.mask[:0])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
