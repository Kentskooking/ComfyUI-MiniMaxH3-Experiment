import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_masked_guider as harness
import test_regional_attention as regional_tests
import torch

import comfy.nested_tensor
import comfy.samplers
import comfy.sampler_helpers
from comfy.ldm.minimax.model import PackedLayout
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from comfy_extras.nodes_minimax_h3 import video_latent_t
from h3_guider_test import continuation as c

GuiderNode = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalInteractionGuider"]


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = regional_tests.RegionalAttentionTests()
        self.fixture.setUp()
        self.base = regional_tests.tiny_model()
        device = torch.device(harness.DEVICE)
        self.patcher = ModelPatcher(self.base, device, torch.device("cpu"))
        self.guider = GuiderNode().get_guider(self.patcher, self.fixture.conditions, "frames", 64, .758)[0]
        self.sigmas = torch.tensor([1., .5, 0.])
        self.noise = Noise_RandomNoise(246)
        self.sampler = comfy.samplers.ksampler("euler")
        self.node = c.H3ContinuationSampler()

    def latent(self, frames=56):
        return {"samples": comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, video_latent_t(frames), 4, 6),
            torch.zeros(1, 32, 2, c.audio_tick(frames))))}

    def run_cycles(self, cycles, **kwargs):
        settings = dict(cycles=cycles, save_checkpoints=False)
        settings.update(kwargs)
        return self.node.sample(self.noise, self.guider, self.sampler, self.sigmas,
                                settings.pop("latent_image", self.latent()), **settings)

    @staticmethod
    def fake_sample(guider, noise, latent, sampler, sigmas, denoise_mask=None, **kwargs):
        return latent + noise * .1 if denoise_mask is None else latent + noise * denoise_mask * .1

    def test_one_cycle_matches_original_native_sampler(self):
        latent = self.latent()
        expected = self.guider.sample(self.noise.generate_noise(latent), latent["samples"], self.sampler,
                                     self.sigmas, disable_pbar=True, seed=self.noise.seed)
        output, state, _ = self.run_cycles(1)
        for actual, previous in zip(output["samples"].unbind(), expected.unbind()):
            torch.testing.assert_close(actual, previous.to(actual.device), rtol=0, atol=0)
        self.assertEqual(state["frames"], 56)
        self.assertEqual(self.noise.seed, 246)

    def test_real_native_sampler_three_cycles_protects_history_and_keeps_joint_attention(self):
        calls = []

        def observe(executor, x, timestep, context, transformer_options, **kwargs):
            payload = kwargs["minimax_payload"]
            calls.append((context.shape[1], len(payload["keyframes"]), x[1].shape[-1],
                          kwargs.get("denoise_mask") is not None))
            return executor(x, timestep, context, transformer_options, **kwargs)

        self.guider.model_patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "test_continuation", observe)
        sources = self.guider.sources
        original_options = self.guider.model_options
        with patch("comfy.model_management.unload_all_models", side_effect=AssertionError("unexpected unload")):
            output, state, report = self.run_cycles(3)
        self.assertEqual(calls, [(8, 2, 93, False)] * 2 + [(8, 2, 94, True)] * 2 + [(8, 2, 93, True)] * 2)
        self.assertEqual(state["frames"], 90)
        self.assertEqual(state["completed_cycles"], 3)
        self.assertEqual(output["samples"].tensors[0].shape[2], 27)
        self.assertEqual(output["samples"].tensors[1].shape[-1], 150)
        self.assertTrue(all(torch.isfinite(x).all() for x in output["samples"].tensors))
        self.assertIn("preserved-prefix max error", report)
        self.assertIs(self.guider.sources, sources)
        self.assertIs(self.guider.model_options, original_options)
        self.assertEqual(self.noise.seed, 246)
        self.assertEqual(self.fixture.conditions[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 0)

    def test_fixed_preparation_runs_once_and_execution_cache_is_released(self):
        sessions = []
        enter = c.ContinuationPreparation.__enter__

        def record(session):
            sessions.append(session)
            return enter(session)

        with patch.object(c.ContinuationPreparation, "__enter__", record), patch.object(
                comfy.sampler_helpers, "_prepare_sampling", wraps=comfy.sampler_helpers._prepare_sampling) as prepare:
            _, _, report = self.run_cycles(3)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args[1][-1], 24 * 17 * 4 * 6 + 32 * 2 * 94)
        session = sessions[0]
        self.assertEqual(session.refine_calls, 1)
        self.assertIsNone(session.refined)
        self.assertEqual(session.models, [])
        self.assertNotIn("h3_continuation_prepare", session.patcher.wrappers.get(WrappersMP.PREPARE_SAMPLING, {}))
        self.assertEqual(self.base.diffusion_model.preprocess_text_embeds.__func__,
                         regional_tests.MiniMaxH3Model.preprocess_text_embeds)
        self.assertIn("Model preparations 1", report)
        self.assertIn("text refinements 1", report)

    def test_inference_failure_releases_cache_and_original_guider_still_runs(self):
        calls, sessions = [], []
        enter = c.ContinuationPreparation.__enter__

        def record(session):
            sessions.append(session)
            return enter(session)

        def fail(executor, *args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("inference interruption")
            return executor(*args, **kwargs)

        self.guider.model_patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "test_failure", fail)
        with patch.object(c.ContinuationPreparation, "__enter__", record):
            with self.assertRaisesRegex(RuntimeError, "inference interruption"):
                self.run_cycles(3)
        self.assertIsNone(sessions[0].refined)
        self.assertEqual(sessions[0].models, [])
        self.assertIsNone(self.base.current_patcher)
        latent = self.latent()
        output = self.guider.sample(self.noise.generate_noise(latent), latent["samples"], self.sampler,
                                   self.sigmas, disable_pbar=True, seed=self.noise.seed)
        self.assertTrue(all(torch.isfinite(x).all() for x in output.tensors))

    def test_preparation_rechecks_eviction_and_reserves_audio_rounding(self):
        latent = self.latent()
        session = c.ContinuationPreparation(self.guider.model_patcher, *latent["samples"].tensors, 56)
        size = [100]
        model = SimpleNamespace(model=object(), is_dynamic=lambda: False, loaded_size=lambda: size[0],
                                patches_uuid=session.patches_uuid)
        calls = []

        def prepare(model, shape, conds, **kwargs):
            calls.append(shape)
            return model.model, conds, []

        with patch.object(comfy.model_management, "loaded_models", return_value=[model]) as loaded:
            for _ in range(2):
                session.prepare(prepare, model, (1, 1, 100), {})
            self.assertEqual(len(calls), 1)
            loaded.return_value = []
            session.prepare(prepare, model, (1, 1, 100), {})
            self.assertEqual(len(calls), 2)
            loaded.return_value = [model]
            size[0] = 50
            session.prepare(prepare, model, (1, 1, 100), {})
            self.assertEqual(len(calls), 3)
            model.patches_uuid = 'new weight patches'
            session.refined = torch.ones(1)
            session.prepare(prepare, model, (1, 1, 100), {})
            self.assertEqual(len(calls), 4)
            self.assertIsNone(session.refined)

    def test_extra_models_keep_native_preparation(self):
        # Attached-model loading and cleanup remain owned by the native sampler.
        patcher = self.guider.model_patcher.clone()
        patcher.additional_models["test_attachment"] = []
        session = c.ContinuationPreparation(patcher, *self.latent()["samples"].tensors, 56)
        with session:
            self.assertFalse(session.enabled)
            self.assertNotIn("h3_continuation_prepare", patcher.wrappers.get(WrappersMP.PREPARE_SAMPLING, {}))

    def test_disk_roundtrip_resume_matches_uninterrupted_native_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c.folder_paths, "get_output_directory", return_value=directory):
            _, first_state, report = self.run_cycles(1, save_checkpoints=True, checkpoint_prefix="proof/run")
            files = list(Path(directory).rglob("*.safetensors"))
            self.assertEqual(len(files), 1)
            restored = c.H3LoadContinuationState().load(str(files[0]))[0]
            for a, b in zip(restored["latent"]["samples"].tensors, first_state["latent"]["samples"].tensors):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            resumed, _, _ = self.run_cycles(2, resume_state=restored)
            continuous, _, _ = self.run_cycles(2)
            for a, b in zip(resumed["samples"].tensors, continuous["samples"].tensors):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertIn(str(files[0]), report)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))

    def test_long_209_frame_chain_rounds_audio_on_absolute_clock(self):
        with patch.object(comfy.samplers.CFGGuider, "sample", autospec=True, side_effect=self.fake_sample) as sample:
            output, state, _ = self.run_cycles(20, latent_image=self.latent(209))
        self.assertEqual(sample.call_count, 20)
        self.assertEqual(state["frames"], 209 + 19 * 170)
        self.assertEqual(output["samples"].tensors[1].shape[-1], c.audio_tick(state["frames"]))
        self.assertEqual([call.args[2].tensors[1].shape[-1] for call in sample.call_args_list[:3]], [348, 349, 348])
        self.assertEqual([call.kwargs["seed"] for call in sample.call_args_list[:3]], [246, 247, 248])

    def test_mask_history_and_future_follow_loop_or_hold_without_mutation(self):
        mask = torch.arange(56, dtype=torch.float32)[:, None, None].expand(56, 4, 6)
        original = [{"mask": mask}, {"mask": torch.ones(1, 4, 6)}]
        sources = c.resize_source_masks(original, self.latent()["samples"].tensors[0], 56, "frames")
        looping = c.window_sources(sources, 17, 56, "frames", "loop")
        holding = c.window_sources(sources, 17, 56, "frames", "hold")
        torch.testing.assert_close(looping[0]["mask"][:39], mask[-39:])
        torch.testing.assert_close(looping[0]["mask"][39:], mask[:17])
        torch.testing.assert_close(holding[0]["mask"][39:], mask[-1:].expand(17, -1, -1))
        latent_mask = [{"mask": torch.arange(17, dtype=torch.float32)[:, None, None]}]
        latent_window = c.window_sources(latent_mask, 17, 56, "latent", "loop")[0]["mask"]
        torch.testing.assert_close(latent_window[:12], latent_mask[0]["mask"][-12:])
        self.assertIs(original[0]["mask"], mask)
        self.assertEqual(looping[1]["mask"].shape[0], 1)

    def test_short_masks_hold_before_looping_and_single_cycle_saves_snapped_context(self):
        mask = torch.arange(20, dtype=torch.float32)[:, None, None].expand(20, 4, 6)
        sources = c.resize_source_masks([{"mask": mask}], self.latent()["samples"].tensors[0], 56, "frames")
        next_mask = c.window_sources(sources, 17, 56, "frames", "loop")[0]["mask"]
        torch.testing.assert_close(next_mask[:3], mask[-3:])
        torch.testing.assert_close(next_mask[3:39], mask[-1:].expand(36, -1, -1))
        torch.testing.assert_close(next_mask[39:], mask[:17])
        with patch.object(comfy.samplers.CFGGuider, "sample", autospec=True, side_effect=self.fake_sample):
            _, state, _ = self.run_cycles(1, latent_image=self.latent(209), context_frames=100)
            self.assertEqual(state["context_frames"], 90)
            _, continued, _ = self.run_cycles(2, latent_image=self.latent(209), context_frames=100, resume_state=state)
            self.assertEqual(continued["frames"], 328)

    def test_position_offsets_keep_anchors_and_text_fixed_with_separate_audio_clock(self):
        layout = PackedLayout(8, 17, 4, 6, 94, keyframes=[{"resolved_frame_index": 0, "latent": self.fixture.images[0]}])
        positions = layout.position_ids.clone()
        captured = {}

        def executor(*args, **kwargs):
            captured.update(kwargs)

        c.offset_positions(executor, None, None, None, {}, {"layout": layout}, start_frame=17)
        shifted = captured["minimax_payload"]["layout"]
        for start, stop, kind in layout.segments:
            expected = positions[start:stop].clone()
            if kind == "video":
                expected[:, 0] += 17 * c.FRAME_RESCALE
            elif kind == "audio":
                expected[:, 0] += 28
            torch.testing.assert_close(shifted.position_ids[start:stop], expected, rtol=0, atol=0)
        torch.testing.assert_close(layout.position_ids, positions, rtol=0, atol=0)

    def test_failed_cycle_keeps_only_completed_checkpoint(self):
        attempts = 0

        def fail_second(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("test cancellation")
            return self.fake_sample(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(c.folder_paths, "get_output_directory", return_value=directory):
            with patch.object(comfy.samplers.CFGGuider, "sample", autospec=True, side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "test cancellation"):
                    self.run_cycles(3, save_checkpoints=True)
            files = list(Path(directory).rglob("*.safetensors"))
            self.assertEqual(len(files), 1)
            self.assertEqual(c.H3LoadContinuationState().load(str(files[0]))[0]["completed_cycles"], 1)
            self.assertFalse(list(Path(directory).rglob("*.tmp")))

    def test_changed_protected_prefix_is_not_accepted(self):
        history = self.latent()
        target = c.carry_prefix(history, history, 17, 56, 39)
        target["samples"].tensors[0][:, :, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "protected video prefix"):
            c.append_future(history, target, 39)

    def test_resume_at_completed_count_skips_sampling_and_rejects_wrong_geometry(self):
        with patch.object(comfy.samplers.CFGGuider, "sample", autospec=True, side_effect=self.fake_sample) as sample:
            _, state, _ = self.run_cycles(2)
            sample.reset_mock()
            _, returned, _ = self.run_cycles(2, resume_state=state)
            sample.assert_not_called()
            self.assertIs(returned["latent"], state["latent"])
            broken = copy.copy(state)
            broken["frames"] += 1
            with self.assertRaisesRegex(ValueError, "timeline"):
                self.run_cycles(3, resume_state=broken)

    def test_paths_schedule_and_context_fail_before_sampling(self):
        self.assertEqual(c.fit_context(100, 209), 90)
        self.assertEqual(c.fit_context(999, 209), 192)
        with self.assertRaisesRegex(ValueError, "at least 56"):
            self.run_cycles(2, latent_image=self.latent(22))
        with patch.object(comfy.samplers.CFGGuider, "sample") as sample:
            with self.assertRaisesRegex(ValueError, "inside"):
                self.run_cycles(2, save_checkpoints=True, checkpoint_prefix="../outside")
            with self.assertRaisesRegex(ValueError, "inside"):
                c.H3LoadContinuationState().load("../outside.safetensors")
            sample.assert_not_called()
        self.sigmas = torch.tensor([1., .5])
        with self.assertRaisesRegex(ValueError, "ending at zero"):
            self.run_cycles(2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
