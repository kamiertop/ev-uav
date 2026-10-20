import io
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from utils import args
from utils.r1_precision import HardBackgroundLoss, STCR1PrecisionLoss
from utils.trajectory_risk import STCTrajectoryRiskLoss
from utils.iou_loss import EventIoULoss


class StubSTC(nn.Module):
    def forward(self, voxel, mapping, predictions, labels):
        return predictions.sum() * 0 + 2


def config(**changes):
    values = dict(risk_mode='event', risk_weight=.1, risk_beta=1, risk_bin_ms=50,
                  iou_weight=.1, hardneg_weight=.03, hardneg_ratio=1., hardneg_min=1)
    values.update(changes)
    return Namespace(**values)


class R1PrecisionTest(unittest.TestCase):
    def test_background_selection_and_gradient_direction(self):
        logits = torch.tensor([1., -2., 2., .3], dtype=torch.double, requires_grad=True)
        labels = torch.tensor([1., 0., 0., 0.])
        loss, stats = HardBackgroundLoss(ratio=1, minimum=1)(logits, labels, torch.zeros(4, 4))
        torch.testing.assert_close(loss, F.softplus(logits[2]))
        loss.backward()
        self.assertGreater(logits.grad[2].item(), 0)
        self.assertTrue((logits.grad[[0, 1, 3]] == 0).all())
        self.assertEqual((stats['hardneg_selected'], stats['hardneg_background']), (1, 3))

    def test_gradcheck_away_from_selection_ties(self):
        logits = torch.tensor([.4, -.7, 1.4, .2], dtype=torch.double, requires_grad=True)
        labels = torch.tensor([1., 0., 0., 0.])
        loss = HardBackgroundLoss(ratio=2, minimum=1)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: loss(value, labels, torch.zeros(4, 4))[0], (logits,)))

    def test_sample_isolation_and_empty_groups(self):
        # The foreground-only sample contributes zero but stays in the sample mean.
        locations = torch.zeros(4, 4)
        locations[2:, 0] = 1
        logits = torch.tensor([1., -1., -2., 2.], requires_grad=True)
        labels = torch.tensor([1., 1., 0., 0.])
        loss, stats = HardBackgroundLoss(ratio=3, minimum=1)(logits, labels, locations)
        torch.testing.assert_close(loss, F.softplus(logits[3]) / 2)
        self.assertEqual(stats['hardneg_empty_samples'], 1)
        self.assertEqual(stats['hardneg_selected'], 1)
        loss.backward()
        self.assertTrue((logits.grad[:3] == 0).all())

    def test_minimum_ratio_and_availability_caps(self):
        logits = torch.arange(10, dtype=torch.float32)
        labels = torch.tensor([1.] * 2 + [0.] * 8)
        for ratio, minimum, expected in [(3, 1, 6), (1, 7, 7), (10, 1, 8)]:
            _, stats = HardBackgroundLoss(ratio, minimum)(logits, labels, torch.zeros(10, 4))
            self.assertEqual(stats['hardneg_selected'], expected)

    def test_half_extremes_and_no_events(self):
        criterion = HardBackgroundLoss()
        for target in [torch.zeros(4), torch.ones(4)]:
            logits = torch.tensor([-100., 100., -10., 10.], dtype=torch.float16, requires_grad=True)
            loss, _ = criterion(logits, target, torch.zeros(4, 4))
            loss.backward()
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())
        empty = torch.empty(0, requires_grad=True)
        loss, stats = criterion(empty, empty, torch.empty(0, 4))
        loss.backward()
        self.assertEqual((loss.item(), stats['hardneg_selected']), (0, 0))

    def test_composition_event_mapping_and_exact_r1_reduction(self):
        logits = torch.tensor([.4, -1.3, 1.6], requires_grad=True)
        mapping = torch.tensor([2, 0, 1, 2])
        labels = torch.tensor([0., 1., 0., 0.])
        locations = torch.zeros(4, 4)
        ids = torch.tensor([0, 1, 0, 0])
        inputs = (None, mapping, logits.sigmoid(), labels,
                  {'segmentation_logits': logits}, locations, ids)
        cfg = config()
        base, _ = STCTrajectoryRiskLoss(StubSTC(), cfg)(*inputs)
        overlap = EventIoULoss('jaccard')(logits[mapping], labels, locations)
        negative, _ = HardBackgroundLoss(1, 1)(logits[mapping], labels, locations)
        loss, stats = STCR1PrecisionLoss(StubSTC(), cfg)(*inputs)
        torch.testing.assert_close(loss, base + .1 * overlap + .03 * negative)
        self.assertIn('risk_weighted', stats)
        self.assertIn('hardneg_weighted', stats)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        disabled = config(iou_weight=0, hardneg_weight=0)
        actual, _ = STCR1PrecisionLoss(StubSTC(), disabled)(*inputs)
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
        # All weights zero skips logits entirely and returns STC unchanged.
        disabled.risk_weight = 0
        no_aux = (*inputs[:4], {}, locations, ids)
        value, _ = STCR1PrecisionLoss(StubSTC(), disabled)(*no_aux)
        self.assertEqual(value.item(), 2)

    def test_controls_enable_only_their_intended_terms(self):
        for preset, iou, neg in [('r1_jaccard', .1, 0), ('r1_hardneg', 0, .03), ('r1_precision', .1, .03)]:
            with patch.object(sys, 'argv', ['train.py', '--experiment', preset]):
                cfg = args.parse()
            self.assertEqual((cfg.iou_weight, cfg.hardneg_weight, cfg.risk_mode), (iou, neg, 'event'))
        for flag, value in [('--hardneg_weight', '-1'), ('--hardneg_ratio', 'nan'),
                            ('--hardneg_min', '0'), ('--risk_weight', '-1'),
                            ('--risk_mode', 'term'), ('--iou_mode', 'lovasz')]:
            with patch.object(sys, 'argv', ['train.py', '--experiment', 'r1_precision', flag, value]), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    args.parse()


if __name__ == '__main__':
    unittest.main()
