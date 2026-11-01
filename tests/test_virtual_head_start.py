from types import SimpleNamespace
import unittest

import test_masked_guider as harness
import torch
from comfy.ldm.minimax.model import FRAME_RESCALE, MiniMaxH3Model, PackedLayout, rope_rotation_table
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, _empty_av_latent

HeadStart = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsVirtualHeadStart"]


class VirtualHeadStartTests(unittest.TestCase):
    def conditioning(self, value=2.0, end=242):
        image = torch.full((1, 24, 1, 2, 4), value, dtype=torch.bfloat16)
        mask = torch.arange(22, dtype=torch.float32).view(22, 1, 1) / 21
        return [[torch.full((1, 4, 8), value, dtype=torch.bfloat16), {
            "minimax_token_tags": torch.tensor([0, 0, 1, 1]),
            "minimax_keyframes": [
                {"resolved_frame_index": 0, "latent": image},
                {"resolved_frame_index": end, "latent": image + 1},
            ],
            "mask": mask, "mask_strength": 0.65, "set_area_to_bounds": False,
        }]]

    def test_slider_and_exact_zero_passthrough(self):
        options = HeadStart.INPUT_TYPES()["required"]["head_start_seconds"][1]
        self.assertEqual((options["min"], options["max"], options["default"], options["display"]), (0, 15, 5, "number"))
        source = self.conditioning()
        self.assertIs(HeadStart().apply(source, 0)[0], source)

    def test_combined_branches_preserve_tensors_masks_and_later_anchors(self):
        source = self.conditioning(2) + self.conditioning(4) + self.conditioning(6)
        source += [[torch.zeros(1, 2, 8), {"strength": 0.5}]]
        for seconds in (0.25, 5.0, 15.0):
            with self.subTest(seconds=seconds):
                output = HeadStart().apply(source, seconds)[0]
                self.assertEqual(len(output), len(source))
                for (old_embedding, old), (new_embedding, new) in zip(source[:3], output[:3]):
                    self.assertIs(old_embedding, new_embedding)
                    self.assertIs(new["mask"], old["mask"])
                    self.assertIs(new["minimax_token_tags"], old["minimax_token_tags"])
                    self.assertEqual(new["mask_strength"], old["mask_strength"])
                    first, last = new["minimax_keyframes"]
                    self.assertEqual(first["resolved_frame_index"], -seconds * 24)
                    self.assertIs(first["latent"], old["minimax_keyframes"][0]["latent"])
                    self.assertIs(last, old["minimax_keyframes"][1])
                    self.assertEqual(old["minimax_keyframes"][0]["resolved_frame_index"], 0)
                self.assertIs(output[-1][1], source[-1][1])

    def test_unanchored_input_has_actionable_error_but_zero_is_valid(self):
        for metadata in ({}, {"minimax_refs": [{"kind": "image"}]},
                         {"minimax_keyframes": [{"resolved_frame_index": 0, "audio_latent": torch.zeros(1, 32, 2, 1)}]}):
            source = [[torch.zeros(1, 4, 8), metadata]]
            self.assertIs(HeadStart().apply(source, 0)[0], source)
            with self.assertRaisesRegex(ValueError, "frame-zero image anchor"):
                HeadStart().apply(source, 5)

    def test_native_layout_and_rope_shift_only_start_anchor_without_extra_tokens(self):
        source = self.conditioning()
        source[0][1]["minimax_keyframes"][0]["audio_latent"] = torch.zeros(1, 32, 2, 3)
        rotary = SimpleNamespace(rope=SimpleNamespace(inv_freq=torch.linspace(.01, .1, 16)))
        for length in (124, 243, 362):
            latent, frames = _empty_av_latent(64, 32, length)
            video, audio = latent["samples"].tensors
            self.assertEqual(frames, length)
            base = PackedLayout(4, video.shape[2], 2, 4, audio.shape[-1], keyframes=source[0][1]["minimax_keyframes"])
            for seconds in (0.1, 5., 15.):
                shifted = HeadStart().apply(source, seconds)[0]
                layout = PackedLayout(4, video.shape[2], 2, 4, audio.shape[-1], keyframes=shifted[0][1]["minimax_keyframes"])
                self.assertEqual(layout.seq_len, base.seq_len)
                self.assertEqual(layout.signature, base.signature)
                self.assertEqual(layout.segments, base.segments)
                expected = base.position_ids.clone()
                for segment in (1, 2):
                    start, stop, kind = layout.segments[segment]
                    self.assertIn(kind, ("cond", "cond_audio"))
                    expected[start:stop, 0] -= seconds * 24 * FRAME_RESCALE
                torch.testing.assert_close(layout.position_ids, expected)
                before = rope_rotation_table(MiniMaxH3Model.rope_freqs(rotary, base.position_ids, torch.device("cpu")), torch.float32)
                after = rope_rotation_table(MiniMaxH3Model.rope_freqs(rotary, layout.position_ids, torch.device("cpu")), torch.float32)
                self.assertTrue(torch.isfinite(after).all())
                anchor_start, anchor_stop, _ = layout.segments[1]
                self.assertFalse(torch.equal(before[:, anchor_start:anchor_stop], after[:, anchor_start:anchor_stop]))
                for start, stop, kind in layout.segments:
                    if kind in ("text", "video", "audio"):
                        torch.testing.assert_close(before[:, start:stop], after[:, start:stop], rtol=0, atol=0)

    def test_native_i2v_encoding_keeps_requested_duration_and_last_frame(self):
        class Clip:
            def tokenize(self, prompt, images):
                return len(images)

            def encode_from_tokens_scheduled(self, count):
                return [[torch.ones(1, 4, 8) * count, {}]]

        class Vae:
            def encode(self, frames):
                return torch.ones(1, 24, 1, frames.shape[1] // 16, frames.shape[2] // 16) * frames.mean()

        result = MiniMaxH3ImageToVideo.execute(Clip(), Vae(), "motion", 64, 32, 243,
                                             first_frame=torch.zeros(1, 32, 64, 3),
                                             last_frame=torch.ones(1, 32, 64, 3))
        video, audio = result[1]["samples"].tensors
        shapes = video.shape, audio.shape
        changed = HeadStart().apply(result[0], 15)[0]
        self.assertEqual([kf["resolved_frame_index"] for kf in changed[0][1]["minimax_keyframes"]], [-360, 242])
        self.assertEqual((video.shape, audio.shape), shapes)
        self.assertEqual(video.shape[2], 72)
        self.assertEqual(audio.shape[-1], 405)
        self.assertEqual(video.count_nonzero().item(), 0)
        self.assertEqual(audio.count_nonzero().item(), 0)

    def test_native_masked_sampler_receives_shifted_originals_and_overlap_pairs(self):
        for seconds in (0., 5., 15.):
            fixture = harness.GuiderTests()
            fixture.setUp()
            mask = torch.linspace(0, 1, 22).view(22, 1, 1).expand(22, 2, 4)
            conditions = [fixture.cond(value, weight, minimax_keyframes=[{
                "resolved_frame_index": 0, "latent": fixture.video[:, :, :1] + value,
            }]) for value, weight in ((2, mask), (6, 1-mask), (10, None))]
            changed = HeadStart().apply(conditions, seconds)[0]
            video, audio = fixture.run_sample(changed, blend_mode="embedding_overlap")
            self.assertEqual(video.shape, fixture.video.shape)
            self.assertEqual(audio.shape, fixture.audio.shape)
            seen_values = set()
            for _, payload, shapes in fixture.model.calls:
                self.assertEqual(shapes, fixture.shapes)
                self.assertEqual(payload["keyframes"][0]["resolved_frame_index"], -seconds * 24)
                self.assertEqual(len(payload["cond_video_latents"]), 1)
                seen_values.add(float(payload["cond_video_latents"][0].mean()))
                layout = payload["layout"]
                start, _, _ = next(s for s in layout.segments if s[2] == "cond")
                self.assertAlmostEqual(float(layout.position_ids[start, 0]), 2 - seconds * 24 * FRAME_RESCALE)
            self.assertTrue({4., 6., 8.}.issubset(seen_values))
            self.assertIs(changed[0][1]["mask"], conditions[0][1]["mask"])
            self.assertEqual(conditions[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
