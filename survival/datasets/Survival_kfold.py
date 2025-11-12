import os
import h5py
import pandas as pd

import torch
import torch.utils.data as data


class Dataset_Survival(data.Dataset):
    def __init__(self, pt_roots, h5_roots, excel_file):
        """
        Args:
            root (str): root directory of the dataset
            excel_file (str): excel file with annotations and splits
        """
        self.pt_roots = pt_roots
        self.h5_roots = h5_roots
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        self.data = self.disc_label(self.data)
        label_dist = self.data['label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        # 检查slides是否有后缀，如有则去掉
        self.check_extension()
        #
        row = self.data.iloc[0]
        self.n_features = torch.load(os.path.join(self.pt_roots[row["dataset"]], str(row["slide"]).split("/")[0] + ".pt"), weights_only=True).shape[-1]
        print("[dataset] dataset from %s" % (self.excel_file))
        print("[dataset] number of cases=%d" % (len(self.data)))
        print("[dataset] number of features=%d" % self.n_features)
        

    def check_extension(self):
        def _rm_ext(row):
            slides = str(row["slide"]).split("/")
            new_slides = []
            for slide in slides:
                if len(str(slide).split(".")) != 1:
                    # 去掉后缀名
                    slide = ".".join(str(slide).split(".")[:-1])
                new_slides.append(slide)
            return "/".join(new_slides)
        self.data["slide"] = self.data.apply(_rm_ext, axis=1)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['time (months)'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['time (months)'].max() + eps
        q_bins[0] = rows['time (months)'].min() - eps
        disc_labels, q_bins = pd.cut(rows['time (months)'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'label', disc_labels)
        return rows
    
    def get_split(self, fold=0):
        assert 0 <= fold <= 4, "fold should be in 0 ~ 4"
        splits = self.data["Fold {}".format(fold)].values.tolist()
        split_dict = {}
        for split in set(splits):
            print("split {}: {} cases".format(split, splits.count(split)))
            split_dict[split] = [i for i, x in enumerate(splits) if x == split]
        return split_dict

    def _load_pt_file(self, dataset, case, slides):
        pt_file = []
        # print(os.path.join(self.pt_roots[dataset], slides + ".pt"))
        if len(str(slides).split("/")) == 1:
            if os.path.exists(os.path.join(self.pt_roots[dataset], str(slides) + ".pt")):
                pt_file = [torch.load(os.path.join(self.pt_roots[dataset], str(slides) + ".pt"), weights_only=True)]
        else:
            for slide in str(slides).split("/"):
                if os.path.exists(os.path.join(self.pt_roots[dataset], str(slide) + ".pt")):
                    pt_file.append(torch.load(os.path.join(self.pt_roots[dataset], str(slide) + ".pt"), weights_only=True))
        if len(pt_file) == 0:
            raise ValueError("No slide found for case: %s" % slides)
        # print(slides)
        pt_file = torch.cat(pt_file, dim=0).to(dtype=torch.float32)
        return pt_file

    def _load_h5_file(self, dataset, case, slides):
        h5_file = []
        for slide in str(slides).split("/"):
            if os.path.exists(os.path.join(self.h5_roots[dataset], slide + ".h5")):
                coords = h5py.File(os.path.join(self.h5_roots[dataset], slide + ".h5"), "r")["coords"][:]
                coords = torch.tensor(coords, dtype=torch.float32)
                h5_file.append(coords)
        if len(h5_file) == 0:
            raise ValueError("No h5 file found for case: %s for slides %s" % (case, slides))
        h5_file = torch.cat(h5_file, dim=0)
        return h5_file

    def __getitem__(self, index):
        row = self.data.iloc[index]
        dataset_ = row["dataset"]
        case_, slide_, censor_, time_, label_ = str(row["case"]), str(row["slide"]), 1 - int(row["status"]), row["time (months)"], row["label"]
        pt_file = self._load_pt_file(dataset_, case_, slide_)
        h5_file = self._load_h5_file(dataset_, case_, slide_)
        censor_ = torch.tensor(censor_, dtype=torch.int64)
        time_ = torch.tensor(time_, dtype=torch.float32)
        label_ = torch.tensor(label_, dtype=torch.int64)
        return case_, slide_, pt_file, h5_file, censor_, time_, label_

    def __len__(self):
        return len(self.data)
