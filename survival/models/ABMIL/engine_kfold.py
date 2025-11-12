import os
import pickle
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored


import torch
import torch.nn.functional as F
import numpy as np
import wandb
import json

class Engine(object):
    def __init__(self, args, results_dir, splits, fold=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold = fold
        self.results = {split: {"C-Index": 0.0, "epoch": 0} for split in splits.keys()}
        if "train" in self.results.keys():
            self.results.pop("train")
        self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.filename_best = None
        self.best_epoch = 0
        self.epoch = 0
        self.early_stop = 0

    def learning(self, model, dataloaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            print("=> loading checkpoint '{}'".format(self.args.resume))
            ckpt = torch.load(self.args.resume, weights_only=False)
            if "results" in ckpt.keys():
                for split in self.results.keys():
                    if split in ckpt["results"].keys():
                        self.results[split] = ckpt["results"][split]
            print({k: round(v["C-Index"], 4) for k, v in self.results.items()})
            self.epoch = self.best_epoch = ckpt["epoch"] if "epoch" in ckpt.keys() else ckpt["best_epoch"]
            model.load_state_dict(ckpt["state_dict"])
            print("=> loaded checkpoint (epoch {})".format(self.epoch))
        if self.args.evaluate:
            for split in self.results.keys():
                # 5fold
                if split in ["train"]:
                    continue
                c_index = self.validate(dataloaders[split], model, criterion, status=split)
                self.results[split] = {"C-Index": c_index,
                                       "epoch": self.best_epoch}
            return self.results

        for epoch in range(self.epoch, self.args.num_epoch):
            print("Epoch: {}".format(epoch))
            self.epoch = epoch
            self.train(dataloaders["train"], model, criterion, optimizer)
            # evaluate
            c_index = self.validate(dataloaders["validation"], model, criterion, status="validation")
            if c_index > self.results["validation"]["C-Index"]:
                self.results["validation"] = {"C-Index": c_index,
                                       "epoch": self.epoch}
                #
                self.best_epoch = self.epoch
                self.save_checkpoint(
                    {
                        "epoch": self.best_epoch,
                        "state_dict": model.state_dict(),
                        "results": self.results,
                    }
                )

                print(" *** best model {}".format(self.filename_best))
            for split in self.results.keys():
                if split in ["train"]:
                    continue
                print(" *** best C-Index results on {} split: {} at epoch {}".format(split, self.results[split]["C-Index"], self.best_epoch))
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")
            wandb.log({f"best_{split}_cindex": self.results[split]["C-Index"] for split in self.results.keys() if split not in ["train"]})
        return self.results

    def train(self, data_loader, model, criterion, optimizer):
        print("running train...")
        model.train()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch)) if self.args.tqdm else data_loader
        for batch_idx, (data_ID, data_Slide, data_WSI, data_Coords, data_censor, data_time, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device) if data_WSI.dtype == torch.float32 else data_WSI.to(self.device).float()
            data_Label = data_Label.to(self.device)
            data_censor = data_censor.to(self.device)

            # prediction
            logit = model(data_WSI)
            hazards = torch.sigmoid(logit)
            S = torch.cumprod(1 - hazards, dim=1)

            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_censor)
            
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            
            # results
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_censor.item()
            all_event_times[batch_idx] = data_time
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        wandb.log({"train_loss": loss, "train_cindex": c_index})


    def validate(self, data_loader, model, criterion, status="validation"):
        print("running {}...".format(status))
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="{} epoch {}".format(status, self.epoch)) if self.args.tqdm else data_loader
        for batch_idx, (data_ID, data_Slide, data_WSI, data_Coords, data_censor, data_time, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device) if data_WSI.dtype == torch.float32 else data_WSI.to(self.device).float()
            data_Label = data_Label.to(self.device)
            data_censor = data_censor.to(self.device)
            
            with torch.no_grad():
                logit = model(data_WSI)
                hazards = torch.sigmoid(logit)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_censor)
            
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_censor.item()
            all_event_times[batch_idx] = data_time
            total_loss += loss.item()
        # calculate loss
        loss = total_loss / len(dataloader)

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        wandb.log({f"{status}_loss": loss, f"{status}_cindex": c_index})
        return c_index

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, "model_best_{epoch}.pth.tar".format(epoch=self.best_epoch))
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)
