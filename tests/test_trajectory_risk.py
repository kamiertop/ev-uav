import math
import unittest
from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F

from utils.trajectory_risk import TrajectoryRisk, STCTrajectoryRiskLoss


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


class TrajectoryRiskTest(unittest.TestCase):
    def test_term_matches_dense_reference_and_gradient(self):
        data = inputs([3., -1., -6., 2.], times=[0, 50, 50, 100])
        logits = data[0]
        events = F.softplus(-logits)
        bins = torch.stack((events[0], events[1:3].mean(), events[3]))
        expected = (torch.logsumexp(2 * bins, 0) - math.log(3)) / 2
        actual, stats = TrajectoryRisk(beta=2)(*data)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, logits)[0],
        )
        self.assertEqual(stats['risk_bins'], 3)
        self.assertLessEqual(actual, bins.max())
        self.assertGreaterEqual(actual, bins.mean())

    def test_beta_zero_and_small_beta_limit(self):
        data = inputs([1., -3., 4.], dtype=torch.float32)
        expected, _ = TrajectoryRisk(mode='balanced')(*data)
        for beta in (0., 1e-8, 1e-5):
            actual, _ = TrajectoryRisk(beta=beta)(*data)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_replicating_bin_events_changes_event_control_but_not_balanced(self):
        original = inputs([3., -3.], times=[0, 50])
        repeated = inputs([3., 3., 3., -3.], times=[0, 0, 0, 50])
        for mode in ('balanced', 'term'):
            loss = TrajectoryRisk(mode=mode)
            torch.testing.assert_close(loss(*original)[0], loss(*repeated)[0])
        self.assertNotAlmostEqual(
            TrajectoryRisk(mode='event')(*original)[0].item(),
            TrajectoryRisk(mode='event')(*repeated)[0].item(),
        )

    def test_sample_and_track_hierarchy_with_reused_ids(self):
        data = inputs([2., -2., 4., 4., -5.],
                      samples=[0, 0, 0, 0, 1], ids=[1, 1, 2, 2, 1],
                      times=[0, 50, 0, 0, 0])
        for mode in ('event', 'balanced', 'term'):
            criterion = TrajectoryRisk(mode=mode)
            combined, stats = criterion(*data)
            split = [criterion(*(x[mask] for x in data))[0]
                     for mask in (data[2][:, 0] == 0, data[2][:, 0] == 1)]
            torch.testing.assert_close(combined, torch.stack(split).mean())
            self.assertEqual(stats['risk_tracks'], 3)
        expected = ((F.softplus(-data[0][:2]).mean() + F.softplus(-data[0][2])) / 2
                    + F.softplus(-data[0][4])) / 2
        torch.testing.assert_close(TrajectoryRisk(mode='balanced')(*data)[0], expected)

    def test_empty_foreground_and_unassigned_ids(self):
        for labels, ids in (([0., 0.], [0, 0]), ([1., 1.], [0, 0])):
            data = inputs([1000., -1000.], labels=labels, ids=ids)
            value, stats = TrajectoryRisk()(*data)
            value.backward()
            self.assertEqual(value.item(), 0)
            self.assertEqual(stats['risk_empty_samples'], 1)
            torch.testing.assert_close(data[0].grad, torch.zeros(2, dtype=torch.float64))
        data = inputs([], times=[])
        value, _ = TrajectoryRisk()(*data)
        value.backward()
        self.assertEqual(value.item(), 0)

    def test_background_and_invalid_id_have_zero_auxiliary_gradient(self):
        data = inputs([0., 1., -2.], labels=[1., 0., 1.], ids=[1, 0, 0])
        value, stats = TrajectoryRisk()(*data)
        value.backward()
        self.assertLess(data[0].grad[0], 0)
        torch.testing.assert_close(data[0].grad[1:], torch.zeros(2, dtype=torch.float64))
        self.assertEqual(stats['risk_unassigned_foreground'], 1)

    def test_empty_sample_kept_in_sample_average(self):
        data = inputs([0., 0.], samples=[0, 1], labels=[1., 0.], ids=[1, 0])
        for mode in ('event', 'balanced', 'term'):
            value, stats = TrajectoryRisk(mode=mode)(*data)
            self.assertAlmostEqual(value.item(), math.log(2) / 2)
            self.assertEqual(stats['risk_empty_samples'], 1)

    def test_extremes_single_bin_and_half_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for times in ([0, 50, 100], [0, 0, 0]):
                data = inputs([-10000., 10000., 0.], times=times, dtype=dtype)
                value, stats = TrajectoryRisk(beta=2)(*data)
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(data[0].grad).all())
                self.assertTrue(torch.isfinite(stats['risk_effective_bins']))

    def test_tilt_increases_hard_bin_gradient_share(self):
        data = inputs([2., -2.])
        shares = []
        for beta in (0., 2.):
            value, _ = TrajectoryRisk(beta=beta)(*data)
            grad = torch.autograd.grad(value, data[0])[0].abs()
            shares.append(grad[1] / grad.sum())
        self.assertGreater(shares[1], shares[0])

    def test_permutation_and_time_origin_invariance(self):
        data = inputs([1., -2., 3.], times=[0, 50, 500])
        criterion = TrajectoryRisk()
        expected = criterion(*data)[0]
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(expected, criterion(*(x[permutation] for x in data))[0])
        data[2][:, 3] += 1234
        torch.testing.assert_close(expected, criterion(*data)[0])

    def test_gradcheck(self):
        data = inputs([.2, -.3, 1.], times=[0, 50, 50])
        criterion = TrajectoryRisk(beta=1.3)
        self.assertTrue(torch.autograd.gradcheck(
            lambda logits: criterion(logits, *data[1:])[0], (data[0],),
        ))

    def test_invalid_parameters(self):
        for kwargs in ({'beta': -1}, {'beta': float('nan')},
                       {'bin_ms': 0}, {'bin_ms': float('inf')}, {'mode': 'invalid'}):
            with self.assertRaises(ValueError):
                TrajectoryRisk(**kwargs)


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
        cfg = Namespace(risk_mode='term', risk_weight=.1, risk_beta=1., risk_bin_ms=50.)
        criterion = STCTrajectoryRiskLoss(ProbabilityBCE(), cfg)
        value, stats = criterion(None, mapping, logits.sigmoid(), labels,
                                 {'segmentation_logits': logits[:, None]}, locations, ids)
        expected = F.binary_cross_entropy_with_logits(logits[mapping], labels)
        expected += .1 * TrajectoryRisk()(logits[mapping], labels, locations, ids)[0]
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0], 0)  # Background is still suppressed.
        self.assertIn('risk_weighted', stats)

    def test_zero_weight_exactly_preserves_base_and_skips_auxiliary(self):
        cfg = Namespace(risk_mode='term', risk_weight=0., risk_beta=1., risk_bin_ms=50.)
        criterion = STCTrajectoryRiskLoss(ProbabilityBCE(), cfg)
        probabilities = torch.tensor([.3, .6], requires_grad=True)
        labels = torch.tensor([0., 1.])
        value, _ = criterion(None, torch.arange(2), probabilities, labels, {}, None, None)
        expected = F.binary_cross_entropy(probabilities, labels)
        self.assertEqual(value.item(), expected.item())
        torch.testing.assert_close(torch.autograd.grad(value, probabilities)[0],
                                   torch.autograd.grad(expected, probabilities)[0])


if __name__ == '__main__':
    unittest.main()
