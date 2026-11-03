"""Check remaining-experiment selection without importing CUDA or reading data."""
import subprocess
import sys
import unittest
from pathlib import Path


class AblationSelectionTest(unittest.TestCase):
    def run_plan(self, *flags):
        runner = Path(__file__).resolve().parents[1] / 'scripts/run_iou_ablation.py'
        return subprocess.run([sys.executable, str(runner), '--dry-run', *flags],
                              capture_output=True, text=True)

    def test_run_only_unfinished_r1_groups(self):
        result = self.run_plan('--suite', 'r1_precision', '--experiments',
                               'r1_jaccard', 'r1_hardneg', 'r1_precision')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if ' train.py ' in line]
        self.assertEqual(len(commands), 3)
        self.assertTrue(all('--epochs 50' in line and '--no-test_after_train' in line for line in commands))
        self.assertNotIn('--experiment r1_event ', result.stdout)

    def test_bad_and_duplicate_selections_rejected(self):
        for flags in [('--suite', 'r1_precision', '--experiments', 'a1'),
                      ('--suite', 'r1_precision', '--experiments', 'r1_event', 'r1_event')]:
            self.assertNotEqual(self.run_plan(*flags).returncode, 0)

    def test_original_default_plan_unchanged(self):
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(' train.py ' in line for line in result.stdout.splitlines()), 9)

    def test_sam_controls_have_unique_runs_and_no_test(self):
        result = self.run_plan('--suite', 'r1_sam')
        self.assertEqual(result.returncode, 0, result.stderr)
        import shlex
        commands = [shlex.split(line) for line in result.stdout.splitlines() if ' train.py ' in line]
        self.assertEqual(len(commands), 4)
        runs = [c[c.index('--run_name') + 1] for c in commands]
        self.assertEqual(len(set(runs)), 4)
        combinations = [(c[c.index('--experiment') + 1], float(c[c.index('--sam_rho') + 1])) for c in commands]
        self.assertEqual(set(combinations), {('r1_jaccard', 0), ('r1_jaccard', .05),
                                            ('r1_precision', 0), ('r1_precision', .05)})
        self.assertTrue(all('--no-test_after_train' in c and c[c.index('--epochs') + 1] == '50' for c in commands))
        for radii in [('0.05', '0.05'), ('nan',), ('-1',)]:
            self.assertNotEqual(self.run_plan('--suite', 'r1_sam', '--sam-rhos', *radii).returncode, 0)


if __name__ == '__main__':
    unittest.main()
