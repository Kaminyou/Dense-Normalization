import argparse
import os
import random
import string

import cv2
import yaml

from utils.util import read_yaml_config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def main():
    """
    USAGE
        python3 transfer.py -c config_example.yaml
        or
        python3 transfer.py -c config_example.yaml --skip_cropping
    """
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument("--skip_cropping", action="store_true")
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    # generate a temperate config file
    dir_path = os.path.dirname(os.path.normpath(args.config))
    temp_config_path = os.path.join(dir_path, f"{id_generator()}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    H, W, _ = cv2.imread(config["INFERENCE_SETTING"]["TEST_X"]).shape
    if not args.skip_cropping:
        os.system(
            f"python3 crop.py -i {config['INFERENCE_SETTING']['TEST_X']} "
            f"-o {config['INFERENCE_SETTING']['TEST_DIR_X']} "
            f"--patch_size {config['CROPPING_SETTING']['PATCH_SIZE']} "
            f"--stride {config['CROPPING_SETTING']['PATCH_SIZE']} "
            f"--thumbnail "
            f"--thumbnail_output {config['INFERENCE_SETTING']['TEST_DIR_X']}"
        )
        print("Finish cropping and start inference")
    os.system(f"python3 inference.py --config {temp_config_path}")
    print("Finish inference and start combining images")
    os.system(
        f"python3 combine.py --config {temp_config_path} "
        f"--resize_h {H} --resize_w {W}"
    )
    os.remove(temp_config_path)


if __name__ == "__main__":
    main()
