import sys
import io
import unittest
from unittest.mock import patch

from utils import args


class ExperimentConfigTest(unittest.TestCase):
    def parse(self, *flags):
        with patch.object(sys, 'argv', ['train.py', *flags]):
            return args.parse()

    def test_presets_and_explicit_override(self):
        for name, expected in args.EXPERIMENTS.items():
            cfg = self.parse('--experiment', name)
            for key, value in expected.items():
                self.assertEqual(getattr(cfg, key), value)
        cfg = self.parse('--loss', 'stc_risk', '--experiment', 'a0')
        self.assertEqual(cfg.loss, 'stc_risk')
        self.assertEqual(cfg.patch_attention, 'legacy')

    def test_boolean_flags_and_legacy_values(self):
        self.assertFalse(self.parse().test_after_train)
        self.assertTrue(self.parse('--test_after_train').test_after_train)
        self.assertFalse(self.parse('--eval', 'False').eval)
        self.assertFalse(self.parse('--no-use_coords').use_coords)
        self.assertTrue(self.parse('--eval').eval)

    def test_current_defaults_and_presets(self):
        cfg = self.parse()
        self.assertEqual((cfg.patch_attention, cfg.loss), ('sequence', 'stc'))
        self.assertEqual(set(args.EXPERIMENTS), {
            'a0', 'attention_fix', 'a1', 'r1_event', 'a1_jaccard',
            'r1_jaccard', 'r1_hardneg', 'r1_precision',
        })

    def test_retired_routes_and_parameters_are_rejected(self):
        for preset in ['r2_balanced', 'r3_term', 'a1_lovasz']:
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse('--experiment', preset)
        for flag in ['--risk_mode', '--risk_beta', '--risk_bin_ms', '--sam_rho',
                     '--sam_start_epoch', '--iou_mode', '--vis', '--save']:
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse(flag, '0')
        cfg = self.parse('--experiment', 'r1_precision')
        for key in ['risk_mode', 'risk_beta', 'risk_bin_ms', 'sam_rho', 'sam_start_epoch', 'iou_mode']:
            self.assertFalse(hasattr(cfg, key))

    def test_retained_weight_overrides_and_invalid_values(self):
        self.assertEqual(self.parse('--experiment', 'r1_precision', '--risk_weight', '0.03').risk_weight, .03)
        for value in ['-1', 'nan', 'inf']:
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse('--experiment', 'r1_precision', '--risk_weight', value)


if __name__ == '__main__':
    unittest.main()
