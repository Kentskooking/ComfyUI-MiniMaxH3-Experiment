import unittest
from unittest.mock import patch

import test_masked_guider as harness
import test_regional_interaction as interaction_tests
import torch

import comfy.model_management
from comfy.ldm.modules.attention import attention_pytorch
from h3_guider_test.regional_attention import RegionalRouting


class RegionalCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = interaction_tests.RegionalInteractionTests()
        self.fixture.setUp()

    def test_reuse_matches_uncached_attention_with_changing_queries_and_values(self):
        preferred = type("Preferred", (), {"function": staticmethod(attention_pytorch)})()
        for conditions in (self.fixture.fixture.conditions, self.fixture.fixture.five_conditions()):
            for interaction in (0., .758, 1.):
                route, layout = self.fixture.routing(interaction, conditions)
                route.cache_limit = 1 << 20
                chunks = (layout.seq_len + route.query_chunk - 1) // route.query_chunk
                with patch.object(route, "bias", wraps=route.bias) as build:
                    for step in range(3):
                        q, k, v = (torch.randn(1, 2, layout.seq_len, 16) for _ in range(3))
                        options = {"minimax_h3_layout": layout}
                        actual = route.attention(None, q, k, v, 2, preferred=preferred,
                                                 transformer_options=options, skip_reshape=True)
                        old, _ = self.fixture.routing(interaction, conditions)
                        expected = old.attention(None, q, k, v, 2, preferred=preferred,
                                                 transformer_options=options, skip_reshape=True)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(build.call_count, chunks)
                    self.assertEqual(route.cache_hits, 2 * chunks)
                self.assertEqual(route.cache_bytes, layout.seq_len ** 2 * 4)
                route.clear_cache()
                self.assertEqual(route.bias_cache, {})
                self.assertEqual(route.cache_bytes, 0)

    def test_partial_budget_recomputes_excess_without_eviction_or_changed_values(self):
        route, layout = self.fixture.routing(.758)
        size = 7 * layout.seq_len * 4
        route.cache_limit = size
        with patch.object(route, "bias", wraps=route.bias) as build:
            first = route.cached_bias(0, 7, torch.device("cpu"), torch.float32)
            for _ in range(3):
                self.assertIs(route.cached_bias(0, 7, torch.device("cpu"), torch.float32), first)
                torch.testing.assert_close(route.cached_bias(7, 14, torch.device("cpu"), torch.float32),
                                           route.bias(7, 14, "cpu", torch.float32), rtol=0, atol=0)
            self.assertEqual(build.call_count, 7)
        self.assertEqual(route.cache_bytes, size)
        self.assertEqual(len(route.bias_cache), 1)

    def test_dtype_and_device_are_part_of_cache_key(self):
        route, layout = self.fixture.routing(.5)
        route.cache_limit = 1 << 20
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            first = route.cached_bias(0, 7, torch.device("cpu"), dtype)
            self.assertIs(route.cached_bias(0, 7, torch.device("cpu"), dtype), first)
            self.assertEqual(first.dtype, dtype)
            torch.testing.assert_close(first, route.bias(0, 7, "cpu", dtype), rtol=0, atol=0)
        self.assertEqual(len(route.bias_cache), 3)
        self.assertEqual(route.cache_bytes, 7 * layout.seq_len * 8)

    def test_budget_respects_free_memory_reserve_and_zero_disables_it(self):
        args = ([{"mask": None}], (3,), [], (1, 24, 2, 4, 6), torch.device("cpu"), "latent", 7)
        with patch.object(comfy.model_management, "minimum_inference_memory", return_value=1000), patch.object(
                comfy.model_management, "get_free_memory", return_value=3000) as free:
            route = RegionalRouting(*args, mask_cache_mb=256)
            self.assertEqual(route.cache_limit, 500)
            free.return_value = 500
            self.assertEqual(RegionalRouting(*args, mask_cache_mb=256).cache_limit, 0)
            free.reset_mock()
            self.assertEqual(RegionalRouting(*args, mask_cache_mb=0).cache_limit, 0)
            free.assert_not_called()

    def test_optional_control_keeps_existing_widget_order(self):
        original = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalAttentionGuider"]
        interaction = harness.pack.NODE_CLASS_MAPPINGS["H3ExperimentsRegionalInteractionGuider"]
        self.assertEqual(list(original.INPUT_TYPES()["required"]), ["model", "conditioning", "mask_space", "query_chunk"])
        self.assertEqual(list(interaction.INPUT_TYPES()["required"]),
                         ["model", "conditioning", "mask_space", "query_chunk", "interaction"])
        self.assertEqual(interaction.INPUT_TYPES()["optional"]["mask_cache_mb"][1]["default"], 256)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
