"""Regression check for launching the runner from scripts/ or another cwd."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunnerImportPathTest(unittest.TestCase):
    def test_root_extension_discovery_from_outside_project(self):
        root = Path(__file__).resolve().parents[1]
        # Run without main/CUDA; a marker module tests root-level discovery
        # independently of whether this machine has a compiled HAIS_OP binary.
        probe = (
            'import runpy,sys,importlib.util; '
            'from pathlib import Path; '
            'sys.path=[p for p in sys.path if p not in ("",sys.argv[2])]; '
            'runpy.run_path(sys.argv[1],run_name="path_probe"); '
            'spec=importlib.util.find_spec("train"); '
            'assert spec is not None; '
            'assert Path(spec.origin).resolve()==Path(sys.argv[2])/"train.py"'
        )
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, '-c', probe,
                            str(root / 'scripts/run_iou_ablation.py'), str(root)],
                           cwd=directory, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
