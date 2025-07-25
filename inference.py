"""
Module: inference.py
Author: Arash Khoeini

Contains scripts to run the model.

Usage:
    python inference.py -m <model_path> -b <batch_size> -i <input_directory> -o <output_directory>

Args:
    -m, --model: path to the trained model (.pth file). Supports models trained using MoCo (v1) and PxCL (v2).
    -i, --input: path to the directory of input images. Input images should be stored directly in this directory, without subdirectories.
    -o, --output: path to the directory where output masks and damage predictions will be stored. If the directory doesn't exist, it will be created automatically.
    -b, --batch: batch size. This is the number of our images that we run through the model simultaneously. Larger the batch size, faster the inference, and larger memory consumption.

Example:
    python inference.py -m path/to/model.pth -b batch_size -i path/to/input/directory -o path/to/output/directory
"""

import argparse
import torch
import numpy as np
from models.unet import UNetVGG16MoCo, UNetVGG16PxCL
from data.plant_loader import InferenceLoader
from tqdm import tqdm
from torchvision import transforms
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from torchvision.utils import make_grid, save_image
from pathlib import Path
import pandas as pd
import time
from memory_profiler import memory_usage
from collections import OrderedDict


def main(model_path, input_path, device, batch_size=1, output_path=None):
    """
    Run inference on plant images using a trained model.

    This function loads a trained model and performs inference on a directory
    of plant images, generating segmentation masks and damage ratio predictions.

    Args:
        model_path (str): Path to the trained model (.pth file)
        input_path (str): Path to directory containing input images
        device (str): Device to run inference on ('cpu', 'cuda', etc.)
        batch_size (int): Batch size for inference (default: 1)
        output_path (str, optional): Output directory for results. If None,
                                   creates 'results' subdirectory in input_path

    Returns:
        None: Results are saved to output directory including:
              - Segmentation masks
              - Colorized visualizations
              - Damage ratio predictions in CSV format
    """

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path / "results"
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    if checkpoint["config"]["pretraining"]["loss"] == "MoCo":
        model = UNetVGG16MoCo(3, pretrained=False)
    elif checkpoint["config"]["pretraining"]["loss"] == "PxCL":
        model = UNetVGG16PxCL(3, pretrained=False)

    key_mapping = {}
    for key in checkpoint["state_dict"]:
        if key.startswith("module"):
            key_mapping[key] = key[key.index("module.") + 7 :]
    checkpoint["state_dict"] = OrderedDict(
        {key_mapping[key]: checkpoint["state_dict"][key] for key in key_mapping.keys()}
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # test_loader = InferenceLoader(data_dir=input_path, batch_size=batch_size, crop_size= 480, num_classes=3, num_workers=1, return_id=True)
    test_loader = InferenceLoader(
        data_dir=input_path, batch_size=batch_size, crop_size=480, num_workers=1
    )

    restore_transform = transforms.Compose(
        [
            local_transforms.DeNormalize(test_loader.MEAN, test_loader.STD),
            transforms.ToPILImage(),
        ]
    )
    viz_transform = transforms.Compose(
        [transforms.Resize((400, 400)), transforms.ToTensor()]
    )
    palette = test_loader.dataset.palette  # type: ignore

    tbar = tqdm(test_loader, ncols=130, disable=False)
    result = []
    for batch_idx, (image, image_id, image_path) in enumerate(tbar):
        image = image.to(device)

        batch_output = model(image)
        for output in batch_output:
            output = output.unsqueeze(0)
            output_np = output.data.max(1)[1].cpu().numpy()

            n_leaf_pixels = np.sum(output_np[output_np == 1])
            n_damage_pixels = np.sum(output_np[output_np == 2])
            damage_ratio = n_damage_pixels / (n_damage_pixels + n_leaf_pixels)

            # creating result grid
            output_np[output_np == 1] = 255
            output_np[output_np == 2] = 120
            d, o = image[0].data.cpu(), output_np[0]
            d = restore_transform(d)
            o = colorize_mask(o, palette)
            d, o = d.convert("RGB"), o.convert("RGB")
            [d, o] = [viz_transform(x) for x in [d, o]]
            val_img = [d, o]

            result_img = make_grid(val_img, nrow=2, padding=5)

            save_image(result_img, output_path / f"{image_id[0]}.png")
            result.append({"id": image_id[0], "damage_ratio": damage_ratio})

    pd.DataFrame(result).to_csv(output_path / "result.txt", index=False)


def f():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="path of trained model (.pth file)",
    )
    parser.add_argument(
        "-i", "--input", default=None, type=str, help="path of input images"
    )
    parser.add_argument(
        "-o", "--output", default=None, type=str, help="path of to write results"
    )
    parser.add_argument(
        "-d", "--device", default="cpu", type=str, help="path of to write results"
    )
    parser.add_argument("-b", "--batch", default=1, type=int, help="batch size")
    args = parser.parse_args()
    main(args.model, args.input, args.device, args.batch, args.output)


if __name__ == "__main__":
    time_start = time.perf_counter()

    mem_usage = memory_usage(f)  # type: ignore

    time_elapsed = time.perf_counter() - time_start
    print("%5.1f secs" % (time_elapsed))
    print("Maximum memory usage: %s" % max(mem_usage))
