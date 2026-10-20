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
        self.assertFalse(self.parse('--vis', 'False').vis)
        self.assertFalse(self.parse('--no-use_coords').use_coords)
        self.assertTrue(self.parse('--vis').vis)

    def test_current_defaults_and_presets(self):
        cfg = self.parse()
        self.assertEqual((cfg.patch_attention, cfg.loss), ('sequence', 'stc'))
        self.assertEqual(set(args.EXPERIMENTS), {
            'a0', 'attention_fix', 'a1', 'r1_event', 'r2_balanced', 'r3_term',
            'a1_lovasz', 'a1_jaccard',
        })

    def test_risk_overrides_and_separate_evaluation_bins(self):
        cfg = self.parse('--risk_mode', 'event', '--experiment', 'r3_term',
                         '--risk_bin_ms', '25', '--risk_beta', '0')
        self.assertEqual(cfg.risk_mode, 'event')
        self.assertEqual(cfg.risk_beta, 0)
        self.assertEqual(cfg.risk_bin_ms, 25)
        self.assertEqual(cfg.trajectory_bin_ms, 50)

    def test_reject_invalid_risk_config(self):
        for flag, value in (('--risk_weight', '-1'), ('--risk_beta', 'nan'),
                            ('--risk_bin_ms', '0')):
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse('--experiment', 'r3_term', flag, value)


if __name__ == '__main__':
    unittest.main()
