import argparse

def get_wandb_config(args):
    wandb_config = dict()
    wandb_config['project'] = args.wandb_proj_name
    wandb_config['name'] = args.wandb_exp_name
    wandb_config['entity'] = args.wandb_entity
    wandb_config['group'] = args.wandb_group
    wandb_config['notes'] = args.wandb_notes
    wandb_config['mode'] = args.wandb_mode
    wandb_config['id'] = args.wandb_id

    return wandb_config

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file")
    parser.add_argument("--feature", type=str, help="which feature extractor to use")
    parser.add_argument("--folds", type=int, default=5, help="which fold to use")
    parser.add_argument("--k_start", type=int, default=-1, help="start fold")
    parser.add_argument("--k_end", type=int, default=-1, help="end fold")
    parser.add_argument("--pt_roots", type=str, help="root directories for pt files.")
    parser.add_argument("--h5_roots", type=str, help="root directories for h5 files.")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--into_memory", type=str, default="True", help="load dataset into memory")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="ABMIL", help="type of model")
    parser.add_argument("--study", type=str, help="used dataset")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function (default: ce)")
    
    # Wandb Parameters
    parser.add_argument("--wandb_proj_name", type=str, default='no-specific-proj', help="wandb_project name")
    parser.add_argument("--wandb_exp_name", type=str, help="wandb_experiment name")
    parser.add_argument("--wandb_entity", type=str, default="your_account", help="wandb entity")
    parser.add_argument("--wandb_tags", type=str, nargs="+", help="wandb tags")
    parser.add_argument("--wandb_group", type=str, help="wandb group")
    parser.add_argument("--wandb_notes", default="", type=str, help="wandb notes")
    parser.add_argument("--wandb_mode", type=str, help="wandb mode")
    parser.add_argument("--wandb_id", type=str, help="wandb id")

    args = parser.parse_args()
    return args
