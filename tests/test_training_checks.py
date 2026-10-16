import unittest
import torch
from utils.training_checks import check_loss, check_gradients, check_tensors


class TrainingChecksTest(unittest.TestCase):
    def test_named_tensor_diagnostic(self):
        check_tensors([("missing", None), ("indices", torch.tensor([1]))])
        with self.assertRaisesRegex(FloatingPointError, "motion"):
            check_tensors([("motion", torch.tensor(float("nan")))])

    def test_reject_nonfinite_loss(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(FloatingPointError):
                check_loss(torch.tensor(value))

    def test_reject_nonfinite_gradient_with_finite_loss(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        loss = parameter.sqrt()
        check_loss(loss)
        loss.backward()
        with self.assertRaises(RuntimeError):
            check_gradients([parameter])

    def test_optional_clipping_and_no_clip_default(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        parameter.grad = torch.tensor([3., 4.])
        check_gradients([parameter])
        torch.testing.assert_close(parameter.grad, torch.tensor([3., 4.]))
        check_gradients([parameter], 1.0)
        self.assertLessEqual(float(parameter.grad.norm()), 1.000001)
