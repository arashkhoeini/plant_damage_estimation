import yaml
from easydict import EasyDict as edict
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict
import argparse


def init_config(config_path: str, argvs: argparse.Namespace) -> edict:
    """
    Initialize configuration from YAML file and command line arguments.

    Args:
        config_path (str): Path to the YAML configuration file
        argvs (argparse.Namespace): Command line arguments

    Returns:
        edict: Configuration dictionary with easy access syntax
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    config = easy_dic(config)
    config = config_parser(config, argvs)
    return config


def easy_dic(dic):
    """
    Convert dictionary to EasyDict for easier access with dot notation.

    Args:
        dic (dict): Input dictionary

    Returns:
        edict: EasyDict version of input dictionary
    """
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic


def config_parser(config, args):
    """
    Update configuration with command line arguments.

    Args:
        config (edict): Configuration dictionary
        args (argparse.Namespace): Command line arguments

    Returns:
        edict: Updated configuration
    """
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    return config


def type_align(source, target):
    """
    Align target type with source type for configuration parsing.

    Args:
        source: Source value with desired type
        target: Target value to convert

    Returns:
        Converted target value with same type as source

    Raises:
        ValueError: If conversion is not supported
    """
    if isinstance(source, bool):
        if target == "False":
            return False
        elif target == "True":
            return True
        else:
            raise ValueError
    elif isinstance(source, float):
        return float(target)
    elif isinstance(source, str):
        return target
    elif isinstance(source, int):
        return int(target)
    elif isinstance(source, list):
        return eval(target)
    else:
        print("Unsupported type: {}".format(type(source)))


def show_config(config, sub=False):
    """
    Display configuration in a formatted way.

    Args:
        config (edict): Configuration to display
        sub (bool): Whether this is a sub-configuration

    Returns:
        str: Formatted configuration string
    """
    msg = ""
    for key, value in config.items():
        if (key == "source") or (key == "target"):
            continue
        if isinstance(value, list):
            value = " ".join([str(v) for v in value])
            msg += "{:>25} : {:<15}\n".format(key, value)
        elif isinstance(value, dict):
            msg += show_config(value, sub=True)
        else:
            msg += "{:>25} : {:<15}\n".format(key, value)
    return msg
