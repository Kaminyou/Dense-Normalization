# Copyright (c) 2024 Ming-Yang Ho
# All rights reserved.
#
# This source code is licensed under the AGPL License found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.model import get_model
from utils.dataset import XInferenceDataset, XPrefetchInferenceDataset
from utils.util import (read_yaml_config, reverse_image_normalize,
                        test_transforms)

MARGIN_PADDING = 16


def main():
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        normalization=config["INFERENCE_SETTING"]["NORMALIZATION"],
        isTrain=False,
        parallelism=config["INFERENCE_SETTING"].get('PARALLELISM', False),
    )

    if config["INFERENCE_SETTING"]["NORMALIZATION"] != 'dn':
        raise ValueError('This normalization method is not supported')

    if config["INFERENCE_SETTING"].get('PARALLELISM', False):
        test_dataset = XPrefetchInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
            pad=MARGIN_PADDING,
        )
    else:
        test_dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
            pad=MARGIN_PADDING,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True,
    )

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    basename = os.path.basename(config["INFERENCE_SETTING"]["TEST_X"])
    filename = os.path.splitext(basename)[0]
    save_path_root = os.path.join(
        config["EXPERIMENT_ROOT_PATH"],
        config["EXPERIMENT_NAME"],
        "test",
        filename,
    )

    if (
        "OVERWRITE_OUTPUT_PATH" in config["INFERENCE_SETTING"]
        and config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"] != ""
    ):
        save_path_root = config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"]

    save_path_base = os.path.join(
        save_path_root,
        config["INFERENCE_SETTING"]["NORMALIZATION"],
        config["INFERENCE_SETTING"]["MODEL_VERSION"],
    )
    os.makedirs(save_path_base, exist_ok=True)
    print(save_path_base)

    if not config["INFERENCE_SETTING"].get('PARALLELISM', False):
        os.makedirs(save_path_base, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N,
        # anchor_num = N but it actually has N + 1 values
        model.init_dense_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1,
            x_anchor_num=x_anchor_num + 1,
        )
        for idx, data in enumerate(test_loader):
            print(f"Caching {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            _ = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=1,
            )

        model.use_dense_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            Y_fake = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=1,
            )
            Y_fake = Y_fake[:, :, MARGIN_PADDING:512 + MARGIN_PADDING, MARGIN_PADDING:512 + MARGIN_PADDING]  # noqa
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_X_{idx}.png",
                    ),
                )
            save_image(
                reverse_image_normalize(Y_fake),
                os.path.join(
                    save_path_base,
                    f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
                ),
            )

    else:
        os.makedirs(save_path_base, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N,
        # anchor_num = N but it actually has N + 1 values
        model.init_prefetch_dense_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1,
            x_anchor_num=x_anchor_num + 1,
        )
        for idx, data in enumerate(test_loader):
            print(f"Executing {idx}", end="\r")

            images = [data['X_img']] + data['pre_img']
            X = torch.cat(images, dim=0)

            Y_fake = model.inference_with_anchor(
                X,
                y_anchor=int(data['y_idx'][0]),
                x_anchor=int(data['x_idx'][0]),
                padding=1,
                pre_y_anchor=[int(i) for i in data['pre_y_idx']],
                pre_x_anchor=[int(i) for i in data['pre_x_idx']],
            )
            Y_fake = Y_fake[[0]]
            Y_fake = Y_fake[:, :, MARGIN_PADDING:512 + MARGIN_PADDING, MARGIN_PADDING:512 + MARGIN_PADDING]  # noqa
            if data['y_idx'][0] != -1:
                X_path = data['X_path']
                save_image(
                    reverse_image_normalize(Y_fake),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
                    ),
                )


if __name__ == "__main__":
    main()
