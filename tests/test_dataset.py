import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset.ev_uav import EvUAV


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / 'train').mkdir()
        self.locations = np.column_stack((np.arange(6), np.arange(6), np.arange(6) * 50))
        self.features = np.column_stack((self.locations / 300., np.ones(6)))
        self.labels = np.arange(6) % 2
        self.ids = self.labels * 3
        np.savez(root / 'train' / 'train_000.npz', ev_loc=self.locations,
                 evs_norm=np.column_stack((self.features, self.labels, self.ids)))
        self.cfg = Namespace(data_dir=str(root), whole_t=8000, res=[346, 260], max_events_num=4)

    def test_sampling_preserves_feature_label_and_identity_alignment(self):
        sample = EvUAV(self.cfg)[0]
        selected = sample['ev_loc'][:, 0].astype(int)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(np.unique(selected)), 4)
        self.assertEqual(set(sample), {'ev_loc', 'evs_norm', 'seg_label', 'idx'})
        np.testing.assert_array_equal(sample['evs_norm'], self.features[selected])
        np.testing.assert_array_equal(sample['seg_label'], self.labels[selected])
        np.testing.assert_array_equal(sample['idx'], self.ids[selected])

    def test_collation_keeps_instance_ids_and_sample_boundaries(self):
        self.cfg.max_events_num = 100
        dataset = EvUAV(self.cfg)
        sample = dataset[0]

        def identity_voxelization(coords, batch_size, mode):
            self.assertEqual(batch_size, 2)
            self.assertEqual(mode, 4)
            n = len(coords)
            return coords, torch.arange(n, dtype=torch.int32), torch.ones(n, 2, dtype=torch.int32)

        # Verify data assembly without requiring the native CUDA voxelizer.
        with patch('dataset.basedataset.voxelization_idx', side_effect=identity_voxelization):
            batch = dataset.custom_collate([sample, sample])
        torch.testing.assert_close(batch['locs'][:, 0], torch.tensor([0] * 6 + [1] * 6))
        torch.testing.assert_close(batch['features'], torch.from_numpy(np.tile(self.features, (2, 1))).float())
        np.testing.assert_array_equal(batch['idx_label'], np.tile(self.ids, 2))
        torch.testing.assert_close(batch['seg_label'], torch.from_numpy(np.tile(self.labels, 2)).to(batch['seg_label'].dtype))


if __name__ == '__main__':
    unittest.main()
