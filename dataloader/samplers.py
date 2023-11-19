from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import itertools
import numpy as np


class LabeledBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, batch_size,fixed_order=False):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size
        self.fixed_order = fixed_order

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        if self.fixed_order:
            primary_iter = self.primary_indices
            print(primary_iter)
        else:
            primary_iter = iterate_once(self.primary_indices)

        res =  (
            primary_batch
            for primary_batch in grouper(primary_iter, self.primary_batch_size)
        )
        # import pdb
        # pdb.set_trace()
        # for secondary_batch in grouper(secondary_iter, self.secondary_batch_size):
        #     print(secondary_batch)
        # if self.fixed_order:
        #     for _ in range(8):
        #         print(next(res))
        return res


    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class UnlabeledBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, secondary_indices, secondary_batch_size,fixed_order=False):
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.fixed_order = fixed_order

        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        if self.fixed_order:
            secondary_iter = self.secondary_indices
            print(secondary_iter)
        else:
            secondary_iter = iterate_eternally(self.secondary_indices)

        res =  (secondary_batch
            for secondary_batch in grouper(secondary_iter, self.secondary_batch_size)
        )
        # import pdb
        # pdb.set_trace()
        # for secondary_batch in grouper(secondary_iter, self.secondary_batch_size):
        #     print(secondary_batch)
        # if self.fixed_order:
        #     for _ in range(8):
        #         print(next(res))
        return res

def iterate_once(iterable):
    return np.random.permutation(iterable)
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
