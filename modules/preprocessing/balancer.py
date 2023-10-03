from collections import defaultdict

import psutil
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
        self.label_to_count = defaultdict(int)
        for idx in self.indices:
            label = self._get_label(idx)
            self.label_to_count[label] += 1

        weights = [
            1.0 / self.label_to_count[self._get_label(idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

    def _get_label(self, idx):
        return self.dataset[idx][1]


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index], self.targets.iloc[index]


class BatchSizeHeuristic:

    @staticmethod
    def estimate(memory_usage_pct, dataset):
        sample_size = len(dataset)
        input_batch, target_batch = next(
            iter(DataLoader(dataset, batch_size=1, shuffle=True)))

        input_size = input_batch.element_size() * input_batch.nelement()
        target_size = target_batch.element_size() * target_batch.nelement()
        batch_memory = input_size + target_size

        available_memory = memory_usage_pct * psutil.virtual_memory().available
        max_batch_size = available_memory // batch_memory
        optimal_batch_size = min(sample_size, max_batch_size)

        return optimal_batch_size
