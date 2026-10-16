import sys
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
        cfg = self.parse('--loss', 'mbtc', '--experiment', 'a0')
        self.assertEqual(cfg.loss, 'mbtc')
        self.assertEqual(cfg.patch_attention, 'legacy')

    def test_boolean_flags_and_legacy_values(self):
        self.assertFalse(self.parse().test_after_train)
        self.assertTrue(self.parse('--test_after_train').test_after_train)
        self.assertFalse(self.parse('--vis', 'False').vis)
        self.assertFalse(self.parse('--no-use_coords').use_coords)
        self.assertTrue(self.parse('--vis').vis)


if __name__ == '__main__':
    unittest.main()
