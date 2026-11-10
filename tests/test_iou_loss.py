import io
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch import nn

from utils import args
from utils.iou_loss import EventIoULoss, STCIoULoss, soft_jaccard
from utils.eval import evalute


class IoULossTest(unittest.TestCase):
    def test_gradients(self):
        logits = torch.tensor([-.7, .3, 1.8, -.2], dtype=torch.double, requires_grad=True)
        labels = torch.tensor([1., 0., 1., 0.], dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(soft_jaccard, (logits, labels)))

    def test_soft_jaccard_formula(self):
        logits = torch.zeros(4)
        labels = torch.tensor([1., 0., 0., 0.])
        self.assertAlmostEqual(soft_jaccard(logits, labels).item(), .8, places=6)

    def test_samples_have_equal_weight_and_permutation_invariance(self):
        logits = torch.tensor([.3, -.4, .8, -1.3])
        labels = torch.tensor([1., 0., 0., 1.])
        locations = torch.zeros(4, 4)
        locations[1:, 0] = 1
        order = torch.tensor([3, 0, 2, 1])
        criterion = EventIoULoss()
        expected = (soft_jaccard(logits[:1], labels[:1]) + soft_jaccard(logits[1:], labels[1:])) / 2
        self.assertTrue(torch.allclose(criterion(logits, labels, locations), expected))
        self.assertTrue(torch.allclose(criterion(logits[order], labels[order], locations[order]), expected))

    def test_empty_and_extreme_half_inputs(self):
        criterion = EventIoULoss()
        empty = torch.empty(0, requires_grad=True)
        criterion(empty, empty, torch.empty(0, 4)).backward()
        for labels in [torch.zeros(4), torch.ones(4)]:
            logits = torch.tensor([-100., -10., 10., 100.], dtype=torch.float16, requires_grad=True)
            loss = criterion(logits, labels, torch.zeros(4, 4))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
        # False foreground in a background-only sample must be pushed down.
        logits = torch.zeros(3, requires_grad=True)
        criterion(logits, torch.zeros(3), torch.zeros(3, 4)).backward()
        self.assertTrue((logits.grad >= 0).all())
        self.assertTrue((logits.grad > 0).any())

    def test_wrapper_maps_events_and_zero_weight_skips_auxiliary(self):
        class StubSTC(nn.Module):
            def forward(self, voxel, mapping, predictions, labels):
                return predictions.sum() * 0 + 2
        logits = torch.tensor([.2, -.3], requires_grad=True)
        mapping = torch.tensor([1, 0, 1])
        labels = torch.tensor([1., 0., 1.])
        locations = torch.zeros(3, 4)
        cfg = Namespace(iou_weight=.1)
        criterion = STCIoULoss(StubSTC(), cfg)
        loss, _ = criterion(None, mapping, logits.sigmoid(), labels,
                            {'segmentation_logits': logits}, locations, None)
        expected = 2 + .1 * soft_jaccard(logits[mapping], labels)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        cfg.iou_weight = 0
        loss, stats = STCIoULoss(StubSTC(), cfg)(None, mapping, logits, labels, {}, locations, None)
        self.assertEqual(loss.item(), 2)
        self.assertEqual(stats['iou_disabled'], 1)

    def test_cli_presets_override_and_reject_bad_weights(self):
        with patch.object(sys, 'argv', ['test', '--experiment', 'a1_jaccard']):
            config = args.parse()
        self.assertEqual((config.loss, config.patch_attention), ('stc_iou', 'sequence'))
        with patch.object(sys, 'argv', ['test', '--experiment', 'a1_jaccard', '--loss', 'stc']):
            self.assertEqual(args.parse().loss, 'stc')
        with patch.object(sys, 'argv', ['test', '--experiment', 'a1_jaccard', '--iou_weight', 'nan']):
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                args.parse()

    def test_global_counts_not_mean_of_sample_ratios(self):
        evaluator = evalute(Namespace(roc=False))
        evaluator.matches = {
            '0': {'seg_pred': torch.tensor([.9, .9, .1]), 'seg_gt': torch.tensor([1., 0., 1.])},
            '1': {'seg_pred': torch.ones(10), 'seg_gt': torch.ones(10)},
        }
        counts = evaluator.evaluate_event_counts(.5)
        self.assertEqual((counts['tp'], counts['fp'], counts['fn']), (11, 1, 1))
        self.assertAlmostEqual(counts['precision'], 11 / 12)
        self.assertAlmostEqual(counts['event_iou'], 11 / 13)
        self.assertAlmostEqual(evaluator.evaluate_semantic_segmantation_miou(.5).item(), 11 / 13, places=6)
        empty = evaluator.event_counts({'seg_pred': torch.ones(2), 'seg_gt': torch.zeros(2)}, .5)
        self.assertIsNone(empty['fp_per_gt'])


if __name__ == '__main__':
    unittest.main()
