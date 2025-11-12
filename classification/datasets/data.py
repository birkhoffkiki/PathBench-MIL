import os
import h5py
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, all_datasets, feature, split_file):
        """
        Args:
            all_datasets (str): excel file with root information of each dataset
            feature (str): feature type, such as "resnet50" or "uni"
            split_file (str): excel file with split information
        """
        self.all_datasets = pd.read_excel(all_datasets)
        self.feature = feature
        self.split_file = pd.read_excel(split_file).reset_index(drop=True)
        print(self.split_file)
        datasets = self.all_datasets["Dataset"].unique()
        datasets = [dataset for dataset in datasets if not pd.isna(dataset)]
        self.roots = {dataset: self.all_datasets[self.all_datasets["Dataset"] == dataset]["Feature Path"].values[0] for dataset in datasets}
        for key, value in self.roots.items():
            if key not in self.split_file["dataset"].unique():
                continue
            if not os.path.exists(value):
                raise ValueError("Feature root %s does not exist" % value)
            else:
                print("[dataset] dataset <%s> from %s" % (key, value))
        # number of classes
        self.classes = self.split_file["label"].unique()
        print(f"[dataset] number of classes={len(self.classes)}: ({self.classes})")
        for label in self.classes:
            print(f"[label] {label}: {self.split_file[self.split_file['label'] == label]['class'].unique()}")
        # number of samples in each split
        self.splits = {split: [i for i, x in enumerate(self.split_file["split"].values.tolist()) if x == split] for split in self.split_file["split"].unique()}
        for key, value in self.splits.items():
            print(f"[dataset] number of cases in {key} split={len(value)}")
        # feature dimension
        try:
            filename = os.path.splitext(self.split_file["slide"].values[0].split("/")[0])[0] + ".pt"
            self.n_features = torch.load(
                os.path.join(self.roots[self.split_file["dataset"].values[0]], "pt_files", self.feature, filename),
                weights_only=True,
            ).shape[-1]
        except:
            raise ValueError("Feature dimension cannot be determined")
        print(f"[dataset] dimension of feature <{self.feature}>={self.n_features}")

    def __load_features__(self, dataset, slides):
        root = self.roots[dataset]
        features = []
        try:
            for slide in slides.split("/"):
                try:
                    features.append(torch.load(os.path.join(root, "pt_files", self.feature, os.path.splitext(slide)[0] + ".pt"), weights_only=False))
                except Exception as e:
                    print(f"[dataset] {slide}: {e}")
                    # raise ValueError(f"[dataset] Error loading features for slide {slide}: {e}")
            if len(features) == 0:
                raise ValueError("[dataset] there is no feature for slide %s" % slides)
            features = torch.cat(features, dim=0)
        except Exception as e:
            print(f"[dataset] cannot loading features for slide {slides}: {e}")
        return features

    def __load_coords__(self, dataset, slides):
        root = self.roots[dataset]
        coords = []
        try:
            for slide in slides.split("/"):
                try:
                    coord = h5py.File(os.path.join(root, "patches", os.path.splitext(slide)[0] + ".h5"), "r")["coords"]
                    coord = np.array(coord)
                    coords.append(torch.tensor(coord))
                except:
                    pass
                    # print(f"[dataset] Cannot load coords for slide {slide}")
            if len(coords) == 0:
                raise ValueError("Cannot load coords for slide %s" % slides)
            coords = torch.cat(coords, dim=0)
        except Exception as e:
            # print(f"[dataset] Error loading coords for slide {slides}: {e}")
            # raise ValueError("Cannot load coords for slide %s" % slides)
            coords = torch.zeros((0, 2))  # Return empty tensor if coords cannot be loaded
        return coords

    def __getitem__(self, idx):
        row = self.split_file.iloc[idx]
        dataset, case, slide, classname, label, split = row["dataset"], row["case"], row["slide"], row["class"], row["label"], row["split"]
        features = self.__load_features__(dataset, slide)
        coords = self.__load_coords__(dataset, slide)
        return dataset, case, slide, classname, features, coords, label, split

    def __len__(self):
        return len(self.split_file)
