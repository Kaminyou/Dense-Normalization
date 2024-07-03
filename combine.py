import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image

from utils.util import read_yaml_config


def main():
    parser = argparse.ArgumentParser("Combined transferred images")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--patch_size", type=int, help="Patch size", default=512
    )
    parser.add_argument(
        "--resize_h", type=int, help="Resize H", default=-1
    )
    parser.add_argument(
        "--resize_w", type=int, help="Resize W", default=-1
    )
    parser.add_argument("--clear_temp_files", action="store_true")
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    basename = os.path.basename(config["INFERENCE_SETTING"]["TEST_X"])
    filename = os.path.splitext(basename)[0]
    path_root = os.path.join(
        config["EXPERIMENT_ROOT_PATH"],
        config["EXPERIMENT_NAME"],
        "test",
        filename,
    )

    if (
        "OVERWRITE_OUTPUT_PATH" in config["INFERENCE_SETTING"]
        and config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"] != ""
    ):
        path_root = config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"]

    path_base = os.path.join(
        path_root,
        config["INFERENCE_SETTING"]["NORMALIZATION"],
        config["INFERENCE_SETTING"]["MODEL_VERSION"],
    )

    combined_image_name = f"combined_" \
                          f"{config['INFERENCE_SETTING']['NORMALIZATION']}_" \
                          f"{config['INFERENCE_SETTING']['MODEL_VERSION']}.png"

    filenames = os.listdir(path_base)
    try:
        filenames.remove('thumbnail_Y_fake.png')
    except Exception:
        pass

    y_anchor_max = 0
    x_anchor_max = 0
    for filename in filenames:
        _, _, y_anchor, x_anchor, _ = filename.split('_', 4)
        y_anchor_max = max(y_anchor_max, int(y_anchor))
        x_anchor_max = max(x_anchor_max, int(x_anchor))

    matrix = np.zeros(
        (y_anchor_max + args.patch_size, x_anchor_max + args.patch_size, 3),
        dtype=np.uint8,
    )

    for filename in sorted(filenames):
        print(f'Combine {filename}  ', end='\r')
        _, _, y_anchor, x_anchor, _ = filename.split('_', 4)
        y_anchor = int(y_anchor)
        x_anchor = int(x_anchor)
        image = cv2.imread(os.path.join(path_base, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        matrix[y_anchor:y_anchor + 512, x_anchor:x_anchor + 512, :] = image

    if (args.resize_h != -1) and (args.resize_w != -1):
        matrix = cv2.resize(
            matrix, (args.resize_w, args.resize_h), cv2.INTER_CUBIC
        )

    matrix_image = Image.fromarray(matrix)
    matrix_image.save(os.path.join(path_root, combined_image_name))

    if args.clear_temp_files:
        shutil.rmtree(path_base)


if __name__ == '__main__':
    main()
