"""Check remaining-experiment selection without importing CUDA or reading data."""
import subprocess
import sys
import unittest
from pathlib import Path


class AblationSelectionTest(unittest.TestCase):
    def run_plan(self, *flags):
        runner = Path(__file__).resolve().parents[1] / 'scripts/run_ablation.py'
        return subprocess.run([sys.executable, str(runner), '--dry-run', *flags],
                              capture_output=True, text=True)

    def test_run_only_unfinished_r1_groups(self):
        result = self.run_plan('--experiments',
                               'r1_jaccard', 'r1_hardneg', 'r1_precision')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if ' train.py ' in line]
        self.assertEqual(len(commands), 3)
        self.assertTrue(all('--epochs 50' in line and '--no-test_after_train' in line for line in commands))
        self.assertNotIn('--experiment r1_event ', result.stdout)

    def test_bad_and_duplicate_selections_rejected(self):
        for flags in [('--experiments', 'r3_term'),
                      ('--experiments', 'r1_event', 'r1_event')]:
            self.assertNotEqual(self.run_plan(*flags).returncode, 0)

    def test_default_four_group_r1_plan(self):
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(' train.py ' in line for line in result.stdout.splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
