import os
import json
import pandas as pd
import time
import warnings


from datasets.Survival_kfold import Dataset_Survival
from utils.options import parse_args
from utils.util import set_seed, CV_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

import wandb
from utils.options import get_wandb_config
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.simplefilter("ignore")


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.resume is not None:
        results_dir = os.path.dirname(os.path.dirname(args.resume.split(";")[0]))
    else:
        results_dir = "results/seed_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print("[log dir] results directory: ", results_dir)
    os.makedirs(results_dir, exist_ok=True)
    # define dataset
    print("[INFO] pt_files: ", args.pt_roots)
    print("[INFO] h5_files: ", args.h5_roots)
    dataset = Dataset_Survival(pt_roots=args.pt_roots, h5_roots=args.h5_roots, excel_file=args.excel_file)
    #
    args.num_classes = 4
    # 5-fold cross validation
    meter = CV_Meter(fold=args.folds)
    if args.k_start == -1:
        args.k_start = 0
    if args.k_end == -1:
        args.k_end = args.folds
    # start 5-fold CV evaluation.
    paths_ckpt = None
    if args.evaluate:
        paths_ckpt = args.resume.split(";")
    for fold in range(args.k_start, args.k_end):
        # get split
        splits =  dataset.get_split(fold)
        dataloaders = {}
        for split in splits:
            dataloaders[split] = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(splits[split]))
        
        if paths_ckpt is not None:
            args.resume = paths_ckpt[fold]
            assert f'fold_{fold}' in args.resume, f"resume path {args.resume} does not match fold {fold}"
        # build model, criterion, optimizer, schedular
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            from models.ABMIL.engine_kfold import Engine

            model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=dataset.n_features)
            engine = Engine(args, results_dir, splits, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)

        scores = engine.learning(model, dataloaders, criterion, optimizer, scheduler)
        meter.updata(scores)
    
    # save results
    meter.save(os.path.join(results_dir, "result.csv"))

if __name__ == "__main__":
    start_time = time.strftime("[%Y-%m-%d]-[%H-%M-%S]")
    print(f"======================================= Start Training at {start_time} =======================================")

    args = parse_args()
    wandb_config = get_wandb_config(args)

    wandb.init(config=args, tags=args.wandb_tags, **wandb_config)
    wandb.config.update(args)
    
    results = main(args)
    print("finished!")
