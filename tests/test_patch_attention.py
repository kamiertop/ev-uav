import unittest

import torch
from model.evspsegnet import patch_attention


class PatchAttentionTest(unittest.TestCase):
    def test_math_backend_matches_explicit_attention_forward_and_backward(self):
        for mode in ("legacy", "corrected", "sequence"):
            module = patch_attention(8, (11, 9, 8), mode=mode, attention_backend="math")
            sequence = torch.randn(3, 1 if mode == "legacy" else 4, 8, requires_grad=True)
            actual = module._attention(sequence)
            expected, _ = module.multihead_attn(sequence, sequence, sequence, need_weights=True)
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual.square().sum(), sequence)[0]
            expected_grad = torch.autograd.grad(expected.square().sum(), sequence)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertTrue(torch.isfinite(actual_grad).all())


    def test_corrected_preserves_legacy_parameters_and_isolates_samples(self):
        torch.manual_seed(4)
        legacy = patch_attention(8, (11, 9, 8), mode="legacy")
        corrected = patch_attention(8, (11, 9, 8), mode="corrected")
        corrected.load_state_dict(legacy.state_dict(), strict=True)
        features = torch.randn(4, 8)
        indices = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0],
                                [1, 0, 0, 0], [1, 1, 0, 0]])
        actual = corrected._attend_features(features, indices, [2, 1, 1])
        expected, _ = legacy.multihead_attn(features[:2].unsqueeze(0),
                                            features[:2].unsqueeze(0),
                                            features[:2].unsqueeze(0))
        torch.testing.assert_close(actual[:2], expected.squeeze(0))
        changed = features.clone()
        changed[2:] += 100
        torch.testing.assert_close(actual[:2], corrected._attend_features(
            changed, indices, [2, 1, 1])[:2])


    def test_samples_do_not_leak_into_each_other(self):
        torch.manual_seed(4)
        module = patch_attention(8, (11, 9, 8), mode="sequence")
        features = torch.randn(7, 8)
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0],
            [1, 0, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0],
        ], dtype=torch.int32)
        original = module._attend_features(features, indices, [4, 2, 2])
        modified = features.clone()
        modified[4:] += 100
        changed = module._attend_features(modified, indices, [4, 2, 2])
        torch.testing.assert_close(original[:4], changed[:4])

    def test_context_changes_a_patch(self):
        torch.manual_seed(5)
        module = patch_attention(8, (11, 9, 8), mode="sequence")
        features = torch.randn(3, 8)
        indices = torch.tensor([
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0],
        ], dtype=torch.int32)
        original = module._attend_features(features, indices, [3, 1, 1])
        modified = features.clone()
        modified[2, 0] += 20
        changed = module._attend_features(modified, indices, [3, 1, 1])
        self.assertGreater((original[0] - changed[0]).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
