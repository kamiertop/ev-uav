import io
import math
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from utils import args
from utils.sharpness import sam_perturbation, batch_stats_without_running_updates


class SharpnessTest(unittest.TestCase):
    def test_quadratic_sam_step_matches_closed_form(self):
        # L=||w||^2/2 => epsilon=rho*w/||w|| and SGD gradient=w+epsilon.
        p = nn.Parameter(torch.tensor([3., 4.], dtype=torch.double))
        optimizer = torch.optim.SGD([p], lr=.1)
        original = p.detach().clone()
        (.5 * p.square().sum()).backward()
        with sam_perturbation([p], .5):
            torch.testing.assert_close(p, original * 1.1)
            optimizer.zero_grad(set_to_none=True)
            (.5 * p.square().sum()).backward()
        torch.testing.assert_close(p, original, rtol=0, atol=0)
        optimizer.step()
        torch.testing.assert_close(p, original - .1 * original * 1.1)

    def test_adam_steps_once_after_exact_restoration(self):
        p = nn.Parameter(torch.tensor([2., -1.], dtype=torch.double))
        original = p.detach().clone()
        optimizer = torch.optim.Adam([p], lr=.01)
        (p.square().sum()).backward()
        with sam_perturbation([p], .05):
            optimizer.zero_grad(set_to_none=True)
            (p.square().sum()).backward()
            selected_gradient = p.grad.clone()
        reference = nn.Parameter(original.clone())
        other = torch.optim.Adam([reference], lr=.01)
        reference.grad = selected_gradient
        optimizer.step()
        other.step()
        torch.testing.assert_close(p, reference, rtol=0, atol=0)
        self.assertEqual(optimizer.state[p]['step'].item(), 1)

    def test_zero_radius_retains_parameters_and_optimizer_result(self):
        p = nn.Parameter(torch.tensor([1., -2.]))
        q = nn.Parameter(p.detach().clone())
        first = torch.optim.Adam([p], lr=.01)
        second = torch.optim.Adam([q], lr=.01)
        p.square().sum().backward()
        q.square().sum().backward()
        with sam_perturbation([p], 0):
            pass
        first.step()
        second.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_failure_restores_weights_and_batchnorm_flags(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).train()
        model(torch.tensor([[1., 2.], [3., 5.]])).square().sum().backward()
        before = [p.detach().clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            with sam_perturbation(model.parameters(), .05):
                with batch_stats_without_running_updates(model):
                    raise RuntimeError('simulated second-pass failure')
        for p, saved in zip(model.parameters(), before):
            torch.testing.assert_close(p, saved, rtol=0, atol=0)
        self.assertTrue(model[1].track_running_stats)

    def test_batchnorm_uses_batch_statistics_but_updates_once(self):
        bn = nn.BatchNorm1d(2, momentum=.1).train()
        x = torch.tensor([[1., 3.], [5., 7.], [2., 4.]])
        first = bn(x)
        saved = {k: v.clone() for k, v in bn.state_dict().items()}
        with batch_stats_without_running_updates(bn):
            second = bn(x)
            second.square().sum().backward()
        torch.testing.assert_close(second, first)
        for key, value in bn.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        self.assertTrue(bn.track_running_stats)

    def test_zero_absent_and_extreme_gradients(self):
        p = nn.Parameter(torch.zeros(2))
        with self.assertRaises(ValueError):
            with sam_perturbation([p], .05):
                pass
        for gradient in [torch.zeros(2), torch.tensor([1e30, -1e30])]:
            p.grad = gradient
            with sam_perturbation([p], .05):
                self.assertTrue(torch.isfinite(p).all())
                expected = 0 if not bool(gradient.any()) else .05
                self.assertAlmostEqual(p.norm().item(), expected, places=6)
            self.assertTrue((p == 0).all())
        p.grad = torch.tensor([float('nan'), 0.])
        with self.assertRaises(FloatingPointError):
            with sam_perturbation([p], .05):
                pass
        self.assertTrue((p == 0).all())

    def test_global_norm_across_parameters_and_frozen_parameters(self):
        first, second = nn.Parameter(torch.tensor([2.])), nn.Parameter(torch.tensor([1.]))
        frozen = nn.Parameter(torch.tensor([5.]), requires_grad=False)
        first.grad, second.grad = torch.tensor([3.]), torch.tensor([4.])
        with sam_perturbation([first, second, frozen], .5):
            self.assertAlmostEqual(first.item(), 2.3, places=6)
            self.assertAlmostEqual(second.item(), 1.4, places=6)
            self.assertEqual(frozen.item(), 5)

    def test_parameter_validation_and_defaults(self):
        with patch.object(sys, 'argv', ['train.py']):
            self.assertEqual(args.parse().sam_rho, 0)
        for flags in [('--sam_rho', '-1'), ('--sam_rho', 'nan'),
                      ('--sam_start_epoch', '-1'),
                      ('--sam_rho', '.05', '--sam_start_epoch', '50')]:
            with patch.object(sys, 'argv', ['train.py', *flags]), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    args.parse()


if __name__ == '__main__':
    unittest.main()
