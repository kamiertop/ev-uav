import math
import unittest
from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from utils.foreground_loss import ForegroundEventLoss, STCForegroundLoss


def inputs(logits, times=None, samples=None, ids=None, labels=None, dtype=torch.float64):
    n = len(logits)
    locations = torch.zeros(n, 4, dtype=torch.long)
    locations[:, 0] = torch.tensor(samples if samples is not None else [0] * n)
    locations[:, 3] = torch.tensor(times if times is not None else list(range(0, 50 * n, 50)))
    return (
        torch.tensor(logits, dtype=dtype, requires_grad=True),
        torch.tensor(labels if labels is not None else [1.] * n),
        locations,
        torch.tensor(ids if ids is not None else [1] * n, dtype=torch.long),
    )


class ForegroundLossTest(unittest.TestCase):
    def test_foreground_mean_and_eligible_ids(self):
        data = inputs([2., -1., 7., -3.], labels=[1., 1., 0., 1.], ids=[1, 2, 0, 0])
        value, stats = ForegroundEventLoss()(*data)
        torch.testing.assert_close(value, F.softplus(-data[0][:2]).mean())
        value.backward()
        self.assertTrue((data[0].grad[:2] < 0).all())
        self.assertTrue((data[0].grad[2:] == 0).all())
        self.assertEqual(stats['risk_unassigned_foreground'], 1)

    def test_samples_equal_weight_including_empty_foreground(self):
        data = inputs([0., 0., 0.], samples=[0, 0, 1], labels=[1., 1., 0.], ids=[1, 2, 0])
        value, stats = ForegroundEventLoss()(*data)
        self.assertAlmostEqual(value.item(), math.log(2) / 2)
        self.assertEqual(stats['risk_empty_samples'], 1)

    def test_event_density_is_not_instance_or_bin_balanced(self):
        data = inputs([3., 3., 3., -3.], times=[0, 0, 0, 50], ids=[1, 1, 1, 2])
        value, _ = ForegroundEventLoss()(*data)
        torch.testing.assert_close(value, F.softplus(-data[0]).mean())
        self.assertNotAlmostEqual(value.item(), F.softplus(-data[0][[0, 3]]).mean().item())

    def test_empty_inputs_and_no_eligible_foreground(self):
        for data in [inputs([]), inputs([1000., -1000.], labels=[0., 0.], ids=[0, 0]),
                     inputs([1000., -1000.], labels=[1., 1.], ids=[0, 0])]:
            value, _ = ForegroundEventLoss()(*data)
            value.backward()
            self.assertEqual(value.item(), 0)
            self.assertTrue((data[0].grad == 0).all())

    def test_extreme_half_inputs_are_finite(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            data = inputs([-10000., 10000., 0.], dtype=dtype)
            value, _ = ForegroundEventLoss()(*data)
            value.backward()
            self.assertTrue(torch.isfinite(value) and torch.isfinite(data[0].grad).all())

    def test_permutation_time_and_positive_instance_id_invariance(self):
        data = inputs([1., -2., 3.], samples=[0, 0, 1], ids=[1, 2, 1])
        expected = ForegroundEventLoss()(*data)[0]
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(expected, ForegroundEventLoss()(*(x[order] for x in data))[0])
        data[2][:, 3] += 9876
        data[3][:] = 99
        torch.testing.assert_close(expected, ForegroundEventLoss()(*data)[0])

    def test_gradcheck_and_bad_alignment(self):
        data = inputs([.2, -.3, 1.])
        self.assertTrue(torch.autograd.gradcheck(lambda logits: ForegroundEventLoss()(logits, *data[1:])[0], (data[0],)))
        with self.assertRaises(ValueError):
            ForegroundEventLoss()(data[0], data[1], torch.zeros(2, 4), data[3])


class ProbabilityBCE(nn.Module):
    """Simple base loss to test composition without CUDA sparse convolutions."""
    def forward(self, voxel, mapping, predictions, labels):
        return F.binary_cross_entropy(predictions.reshape(-1)[mapping], labels)


class RiskIntegrationTest(unittest.TestCase):
    def test_model_checkpoint_keys_and_initial_parameters_match_a1(self):
        from utils import args
        from model.evspsegnet import evspsegnet
        from unittest.mock import patch
        import sys
        with patch.object(sys, 'argv', ['train.py', '--experiment', 'a1']):
            cfg = args.parse()
        torch.manual_seed(37)
        baseline = evspsegnet(cfg)
        cfg.loss = 'stc_risk'
        torch.manual_seed(37)
        candidate = evspsegnet(cfg)
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),
                         sum(p.numel() for p in candidate.parameters()))
        features = torch.randn(5, cfg.width)
        old_probabilities = baseline.semantic_linear(features)
        logits = candidate.semantic_linear[0](features)
        torch.testing.assert_close(old_probabilities, candidate.semantic_linear[1](logits),
                                   rtol=0, atol=0)

    def test_voxel_mapping_and_background_gradient(self):
        logits = torch.tensor([.2, -.3, 1.], dtype=torch.float64, requires_grad=True)
        mapping = torch.tensor([2, 0, 2, 1])
        _, labels, locations, ids = inputs([0.] * 4, labels=[1., 0., 1., 1.], ids=[1, 0, 1, 2])
        labels = labels.double()
        cfg = Namespace(risk_weight=.1)
        criterion = STCForegroundLoss(ProbabilityBCE(), cfg)
        value, stats = criterion(None, mapping, logits.sigmoid(), labels,
                                 {'segmentation_logits': logits[:, None]}, locations, ids)
        expected = F.binary_cross_entropy_with_logits(logits[mapping], labels)
        expected += .1 * ForegroundEventLoss()(logits[mapping], labels, locations, ids)[0]
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0], 0)  # Background is still suppressed.
        self.assertIn('risk_weighted', stats)

    def test_zero_weight_exactly_preserves_base_and_skips_auxiliary(self):
        cfg = Namespace(risk_weight=0.)
        criterion = STCForegroundLoss(ProbabilityBCE(), cfg)
        probabilities = torch.tensor([.3, .6], requires_grad=True)
        labels = torch.tensor([0., 1.])
        value, _ = criterion(None, torch.arange(2), probabilities, labels, {}, None, None)
        expected = F.binary_cross_entropy(probabilities, labels)
        self.assertEqual(value.item(), expected.item())
        torch.testing.assert_close(torch.autograd.grad(value, probabilities)[0],
                                   torch.autograd.grad(expected, probabilities)[0])


if __name__ == '__main__':
    unittest.main()
