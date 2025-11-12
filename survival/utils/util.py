import os
import csv
import random
import numpy as np
import wandb
import torch
from torch.utils.data import Sampler


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)



def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter:
    def __init__(self, fold):
        self.fold = fold
        self.splits = None  # will be set on first updata
        self.header = ["folds"]
        self.rows = []

    def updata(self, results):
        '''
        results: dict
        {'val': {'C-Index': 0.0, 'epoch': 0},
         'split_1': {'C-Index': 0.0, 'epoch': 0},
         'split_2': {'C-Index': 0.0, 'epoch': 0}}
        '''
        if self.splits is None:
            self.splits = list(results.keys())
            # build header: folds, epoch_val, C-Index_val, epoch_split_1, C-Index_split_1, ...
            for split in self.splits:
                self.header.append(f"epoch_{split}")
                self.header.append(f"C-Index_{split}")
        row = [len(self.rows)]
        for split in self.splits:
            row.append(results[split]['epoch'])
            row.append(round(results[split]["C-Index"], 4))
        print(row)
        self.rows.append(row)

    def save(self, path):
        print("save evaluation resluts to", path)
        if self.fold > 1 and self.splits is not None:
            means = ["mean"]
            stds = ["std"]
            # For each split, calculate mean and std for C-Index
            for split_idx, split in enumerate(self.splits):
                cindex_col = 2 + split_idx * 2  # column index for C-Index_{split}
                cindex_values = [r[cindex_col] for r in self.rows]
                mean = round(np.mean(cindex_values), 4)
                std = round(np.std(cindex_values), 4)
                means.extend(["-", mean])
                stds.extend(["-", std])
                wandb.log({f"mean_{split}": mean, f"std_{split}": std})
            self.rows.append(means)
            self.rows.append(stds)
        with open(path, "a", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerows(self.rows)
        # os.chmod(path, 0o777)


if __name__ == "__main__":
    meter = CV_Meter(5)
    for i in range(5):
        meter.updata({
            "val": {"C-Index": i, "epoch": i},
            "split_1": {"C-Index": i+0.1, "epoch": i+1},
            "split_2": {"C-Index": i+0.2, "epoch": i+2},
        })
    meter.save("test.csv")
    
