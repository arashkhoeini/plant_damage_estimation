import os
import argparse
import torch
from data.plant_loader import PlantLoader
from utils import losses
from utils.logger import Logger
from trainer import Trainer
import random
import numpy as np
from configs.init_configs import init_config
from tabulate import tabulate
import pandas as pd
from typing import List, Dict, Any, Union


def init_seed(seed: int) -> None:
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False  # type: ignore
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_instance(module: Any, name: str, config: Dict[str, Any], *args: Any) -> Any:
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


def aggregate_results(
    results: List[Dict[str, Union[float, int]]],
) -> Dict[str, Union[float, int]]:
    """
    aggregates results from different experiments (folds)

    Args:
        results (List[Dictionary])

    Returns:
        averaged results
    """
    keys = results[0].keys()
    num_dicts = len(results)
    avg_results = {}

    for key in keys:
        total = sum([d[key] for d in results])
        avg_results[key] = total / num_dicts

    return avg_results


def write_results(output_dir: str, avg_result: Dict[str, Union[float, int]]) -> None:
    """
    Writes results in tabulated format in {output_dir}/results.txt

    Args:
        output_dir (str): output directory to write results
        avg_results (Dictionary): results to write
    """
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(
            tabulate(
                pd.DataFrame(avg_result, index=[0]), headers="keys", tablefmt="psql"  # type: ignore
            )
        )


def main(config: Dict[str, Any]) -> None:
    train_logger = Logger()

    # LOSS
    pretrain_loss = getattr(losses, config["pretraining"]["loss"])()
    train_loss = getattr(losses, config["finetuning"]["loss"])()

    if config["testing"]["cross_validation"]:
        filename = config[
            "pretrained"
        ]  # path of the pretrained network if exists. Otherwise None
        results = []
        for i in range(1, config["testing"]["n_folds"] + 1):
            train_splits = []

            for j in range(1, config["testing"]["n_folds"] + 1):
                if i == j:
                    test_split = [f"fold{i}"]
                else:
                    train_splits.append(f"fold{j}")

            if config["finetuning"]["val"]:
                val_idx = random.randint(0, len(train_splits) - 1)
                val_split = [train_splits.pop(val_idx)]
            else:
                val_split = test_split

            config["finetune_loader"]["args"]["splits"] = train_splits
            config["val_loader"]["args"]["splits"] = val_split
            config["test_loader"]["args"]["splits"] = test_split

            for loader in [
                "pretrain_loader",
                "finetune_loader",
                "val_loader",
                "test_loader",
            ]:
                config[loader]["args"]["data_dir"] = config["data_dir"]
                config[loader]["args"]["num_classes"] = config["num_classes"]

            # DATA LOADERS
            pretrain_loader = PlantLoader(**config["pretrain_loader"]["args"])
            finetune_loader = PlantLoader(**config["finetune_loader"]["args"])
            val_loader = PlantLoader(**config["val_loader"]["args"])
            test_loader = PlantLoader(**config["test_loader"]["args"])

            # TRAINING
            trainer = Trainer(
                pretrain_loss=pretrain_loss,
                train_loss=train_loss,
                config=config,
                pretrain_loader=pretrain_loader,
                train_loader=finetune_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                train_logger=train_logger,
            )

            # RELOAD PRETRAINED NET IF NOT THE FIRST FOLD
            print(filename)
            if filename and filename != "":
                trainer._resume_checkpoint(filename)
            else:
                filename = trainer.pretrain()
            trainer.train()
            fold_result = trainer.test()
            results.append(fold_result)
        write_results(config["output_dir"], aggregate_results(results))
    else:
        filename = config[
            "pretrained"
        ]  # path of the pretrained network if exists. Otherwise None

        config["train_loader"]["args"]["splits"] = [
            "fold1",
            "fold2",
            "fold3",
            "fold4",
            "fold5",
        ]
        config["val_loader"]["args"]["splits"] = ["fold1"]
        config["test_loader"]["args"]["splits"] = ["fold1"]

        for loader in ["pretrain_loader", "train_loader", "val_loader", "test_loader"]:
            config[loader]["args"]["data_dir"] = config["data_dir"]
            config[loader]["args"]["num_classes"] = config["num_classes"]

        # DATA LOADERS
        pretrain_loader = PlantLoader(**config["pretrain_loader"]["args"])
        train_loader = PlantLoader(**config["train_loader"]["args"])
        val_loader = PlantLoader(**config["val_loader"]["args"])
        test_loader = PlantLoader(**config["test_loader"]["args"])

        # TRAINING
        trainer = Trainer(
            pretrain_loss=pretrain_loss,
            train_loss=train_loss,
            config=config,
            pretrain_loader=pretrain_loader,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_logger=train_logger,
        )

        # RELOAD PRETRAINED NET IF NOT THE FIRST FOLD
        print(filename)
        if filename != "":
            trainer._resume_checkpoint(filename)
        else:
            filename = trainer.pretrain()
        trainer.train()


if __name__ == "__main__":
    # Any args defined here will overwrite the arg with the same name in configs.yml
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed")
    args = parser.parse_args()

    config = init_config("configs/configs.yml", args)

    # if args.resume:
    #     config = torch.load(args.resume)['config']

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    init_seed(args.seed)

    main(config)
