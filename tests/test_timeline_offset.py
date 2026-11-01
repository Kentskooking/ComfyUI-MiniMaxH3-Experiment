from types import SimpleNamespace
import unittest

import test_masked_guider as harness
import torch
import comfy.ops
import comfy.supported_models
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, rope_rotation_table
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

Offset = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsGeneratedTimelineOffset"]
HeadStart = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsVirtualHeadStart"]


class TimelineOffsetTests(unittest.TestCase):
    def setUp(self):
        self.patcher = ModelPatcher(harness.TestH3(), torch.device("cpu"), torch.device("cpu"))

    def payload(self, length=124, keyframes=None, refs=None):
        latent, _ = _empty_av_latent(64, 32, length)
        video, audio = latent["samples"].tensors
        context = torch.ones(1, 4, 8)
        payload = {"keyframes": keyframes, "refs": refs, "seed": 123,
                   "layout": PackedLayout(4, video.shape[2], 2, 4, audio.shape[-1], keyframes=keyframes, refs=refs)}
        return [video, audio], context, payload

    def capture(self, patcher, inputs):
        x, context, payload = inputs
        timestep = torch.tensor([500.])
        mask = torch.ones_like(x[0][:, :1])
        options = {"sentinel": object()}

        def downstream(actual_x, actual_t, actual_context, actual_options, minimax_payload, denoise_mask):
            self.assertIs(actual_x, x)
            self.assertIs(actual_t, timestep)
            self.assertIs(actual_context, context)
            self.assertIs(denoise_mask, mask)
            self.assertEqual(actual_options, options)
            return minimax_payload

        result = WrapperExecutor.new_executor(downstream, patcher.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)).execute(
            x, timestep, context, options, minimax_payload=payload, denoise_mask=mask)
        self.assertNotIn("minimax_h3_layout", options)
        return result

    def test_slider_zero_and_model_clone_isolation(self):
        settings = Offset.INPUT_TYPES()["required"]["forward_seconds"][1]
        self.assertEqual((settings["min"], settings["max"], settings["default"], settings["display"]), (0, 15, 5, "slider"))
        off = Offset().patch(self.patcher, 0)[0]
        on = Offset().patch(self.patcher, 5)[0]
        self.assertIs(on.model, self.patcher.model)
        self.assertEqual(self.patcher.wrappers, {})
        self.assertEqual(off.get_all_wrappers(WrappersMP.DIFFUSION_MODEL), [])
        inputs = self.payload()
        self.assertIs(self.capture(off, inputs), inputs[2])

    def test_offsets_preserve_duration_spacing_spatial_positions_and_conditioning(self):
        image = torch.ones(1, 24, 1, 2, 4)
        for length in (124, 243, 362):
            keyframes = [{"resolved_frame_index": 0, "latent": image, "audio_latent": torch.ones(1, 32, 2, 3)},
                         {"resolved_frame_index": length - 1, "latent": image}]
            inputs = self.payload(length, keyframes=keyframes)
            original = inputs[2]["layout"]
            saved = original.position_ids.clone()
            for seconds in (.1, .25, 5., 15.):
                with self.subTest(length=length, seconds=seconds):
                    patched = Offset().patch(self.patcher, seconds)[0]
                    for _ in range(2):
                        output = self.capture(patched, inputs)
                        shifted = output["layout"]
                        self.assertIs(output["keyframes"], keyframes)
                        self.assertEqual(shifted.signature, original.signature)
                        self.assertEqual(shifted.seq_len, original.seq_len)
                        self.assertIs(shifted.segments, original.segments)
                        self.assertIs(shifted.img_update, original.img_update)
                        self.assertIs(shifted.audio_update, original.audio_update)
                        torch.testing.assert_close(shifted.position_ids[:, 1:], saved[:, 1:], rtol=0, atol=0)
                        for start, stop, kind in original.segments:
                            expected = saved[start:stop, 0] + (seconds * 40 if kind in ("video", "audio") else 0)
                            torch.testing.assert_close(shifted.position_ids[start:stop, 0], expected)
                            torch.testing.assert_close(shifted.position_ids[start:stop, 0].diff(), saved[start:stop, 0].diff())
                    torch.testing.assert_close(original.position_ids, saved, rtol=0, atol=0)

    def test_head_start_and_forward_offset_add_first_image_separation(self):
        anchor = {"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 2, 4)}
        conditioning = [[torch.ones(1, 4, 8), {"minimax_keyframes": [anchor]}]]
        moved = HeadStart().apply(conditioning, 5)[0]
        inputs = self.payload(keyframes=moved[0][1]["minimax_keyframes"])
        layout = self.capture(Offset().patch(self.patcher, 5)[0], inputs)["layout"]
        origins = {kind: float(layout.position_ids[start, 0]) for start, stop, kind in layout.segments}
        self.assertEqual(origins["cond"], 4 - 200)
        self.assertEqual(origins["video"], 4 + 200)
        self.assertEqual(origins["audio"], origins["video"])
        self.assertEqual((origins["video"] - origins["cond"]) / 40, 10)
        self.assertEqual(anchor["resolved_frame_index"], 0)

    def test_omni_references_and_text_visual_tokens_stay_fixed(self):
        refs = [{"kind": "image", "latent_h": 2, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 3},
                {"kind": "video_audio", "latent_t": 2, "latent_h": 2, "latent_w": 4, "ref_audio_t": 8}]
        inputs = self.payload(refs=refs)
        inputs[2]["text_token_tags"] = torch.tensor([0, 0, 1, 1])
        output = self.capture(Offset().patch(self.patcher, 15)[0], inputs)
        self.assertIs(output["refs"], refs)
        self.assertIs(output["text_token_tags"], inputs[2]["text_token_tags"])
        for start, stop, kind in output["layout"].segments:
            if kind not in ("video", "audio"):
                torch.testing.assert_close(output["layout"].position_ids[start:stop], inputs[2]["layout"].position_ids[start:stop], rtol=0, atol=0)

    def test_chaining_replaces_offset_and_zero_removes_only_our_wrapper(self):
        def unrelated(executor, *args, **kwargs):
            return executor(*args, **kwargs)

        self.patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "unrelated", unrelated)
        first = Offset().patch(self.patcher, 5)[0]
        second = Offset().patch(first, 2)[0]
        off = Offset().patch(second, 0)[0]
        self.assertEqual(off.get_all_wrappers(WrappersMP.DIFFUSION_MODEL), [unrelated])
        self.assertEqual(len(second.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)), 2)
        inputs = self.payload()
        for patcher, seconds in ((first, 5), (second, 2), (off, 0)):
            layout = self.capture(patcher, inputs)["layout"]
            start, _, _ = next(s for s in layout.segments if s[2] == "video")
            self.assertEqual(float(layout.position_ids[start, 0]) - float(inputs[2]["layout"].position_ids[start, 0]), seconds * 40)

    def test_missing_native_layout_has_actionable_error(self):
        inputs = self.payload()
        inputs[2].pop("layout")
        with self.assertRaisesRegex(ValueError, "native H3 model"):
            self.capture(Offset().patch(self.patcher, 5)[0], inputs)

    def test_native_rope_changes_only_target_streams(self):
        inputs = self.payload(keyframes=[{"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 2, 4)}])
        original = inputs[2]["layout"]
        shifted = self.capture(Offset().patch(self.patcher, 5)[0], inputs)["layout"]
        rotary = SimpleNamespace(rope=SimpleNamespace(inv_freq=torch.linspace(.01, .1, 16)))
        before = rope_rotation_table(MiniMaxH3Model.rope_freqs(rotary, original.position_ids, torch.device("cpu")), torch.float32)
        after = rope_rotation_table(MiniMaxH3Model.rope_freqs(rotary, shifted.position_ids, torch.device("cpu")), torch.float32)
        self.assertTrue(torch.isfinite(after).all())
        for start, stop, kind in original.segments:
            if kind in ("audio", "video"):
                self.assertFalse(torch.equal(before[:, start:stop], after[:, start:stop]))
            else:
                torch.testing.assert_close(before[:, start:stop], after[:, start:stop], rtol=0, atol=0)

    def test_native_h3_attention_consumes_offset_with_existing_block_patch(self):
        torch.manual_seed(17)
        config = comfy.supported_models.MiniMaxH3(dict(
            hidden_size=32, num_layers=1, token_refiner_num_layers=1, num_attention_heads=1,
            attention_head_dim=32, ffn_hidden_size=64, text_dim=8, timestep_input_dim=8,
            time_embed_hidden_size=32, time_embed_dim=16, rope_inv_freq_len=4, dtype=torch.float32))
        config.custom_operations = comfy.ops.manual_cast
        base = comfy.model_base.MiniMaxH3(config, device=torch.device("cpu"))
        model = base.diffusion_model
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, .05)
        model.rope.inv_freq.copy_(torch.tensor([.1, .03, .01, .003]))
        patcher = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
        video = torch.randn(1, 24, 7, 3, 5)
        audio = torch.randn(1, 32, 2, 37)
        anchor = torch.randn(1, 24, 1, 4, 6)
        conditions = base.extra_conds(
            cross_attn=torch.randn(1, 4, 8), latent_shapes=[video.shape, audio.shape], device=torch.device("cpu"),
            minimax_keyframes=[{"resolved_frame_index": -120, "latent": anchor}])
        context = conditions["c_crossattn"].cond
        payload = conditions["minimax_payload"].cond
        original = payload["layout"].position_ids.clone()
        seen = []

        def existing_block_patch(args, extra):
            seen.append((args["layout"].position_ids.clone(), args["t_emb"].clone()))
            return extra["original_block"](args)

        patcher.set_model_patch_replace(existing_block_patch, "dit", "double_block", 0)
        outputs = []
        for seconds in (0, 5, 15):
            patched = Offset().patch(patcher, seconds)[0]
            options = {**patched.model_options["transformer_options"], "wrappers": patched.wrappers}
            out = model([video, audio], torch.tensor([500.]), context,
                        transformer_options=options, minimax_payload=payload)
            self.assertEqual([z.shape for z in out], [video.shape, audio.shape])
            self.assertTrue(all(torch.isfinite(z).all() for z in out))
            expected = original.clone()
            for start, stop, kind in payload["layout"].segments:
                if kind in ("video", "audio"):
                    expected[start:stop, 0] += seconds * 40
            torch.testing.assert_close(seen[-1][0], expected)
            torch.testing.assert_close(seen[-1][1], seen[0][1], rtol=0, atol=0)
            torch.testing.assert_close(payload["layout"].position_ids, original, rtol=0, atol=0)
            outputs.append(out)
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(outputs[0], outputs[1])))
        self.assertEqual(patcher.get_all_wrappers(WrappersMP.DIFFUSION_MODEL), [])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
