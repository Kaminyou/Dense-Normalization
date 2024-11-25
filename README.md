[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![version](https://img.shields.io/badge/version-1.0.0-red)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://github.com/Kaminyou/Dense-Normalization/blob/main/LICENSE)
![linting workflow](https://github.com/Kaminyou/Dense-Normalization/actions/workflows/main.yml/badge.svg)
<div align="center">

<h2>ECCV 2024</h2>
<h1>Every Pixel Has its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization</h1>

[Ming-Yang Ho](https://kaminyou.com/)<sup>1</sup>, &nbsp; [Che-Ming Wu](https://github.com/st9007a)<sup>2</sup>, &nbsp; [Min-Sheng Wu](https://github.com/Min-Sheng)<sup>3</sup>, &nbsp; [Yufeng Jane Tseng](https://www.csie.ntu.edu.tw/en/member/Faculty/Yufeng-Jane-Tseng-95281407)<sup>1</sup>

<sup>1</sup>National Taiwan University, &nbsp; <sup>2</sup>Amazon Web Services, &nbsp; <sup>3</sup>aetherAI<br>

[[`Paper (arxiv)`](https://arxiv.org/abs/2407.04245)] [[`Paper (official)`](https://link.springer.com/chapter/10.1007/978-3-031-72995-9_18)] [[`Project Page`](https://kaminyou.com/Dense-Normalization/)]
<br><br><image src="./images/teaser.jpg"/>
<br><image src="./images/framework.jpg"/>

</div>

## Get started with an example
We provide a simple example (one image from the Kyoto summer2autumn dataset) for you to translate an UHR image with our DN.

### Download example data
```bash
$ ./download.sh
$ unzip simple_example.zip
```

### Environment preparation
1. Please check your GPU driver version and modify `Dockerifle` accordingly
2. Then, execute
    ```bash
    $ docker-compose up --build -d
    ```
3. Get into the docker container
    ```bash
    $ docker exec -it dn-env bash
    ```

### Inference
1. In the docker container, please execute
    ```bash
    $ python3 transfer.py -c data/japan/config.yaml
    ```
2. Then, you can see a translated image at `experiments/japan_CUT/test/IMG_6610/combined_dn_10.png`
3. To see the image conveniently, you can leverage the provided `visualization.ipynb`. The setup of jupyter notebbok can be achived by
    - a. modify a port mapping setting in `docker-compose.yml`; e,g, `- 19000:8888`
    - b. install `jupyter` in the container
    - c. run your jupyter notebook by `nohup jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &`
    - d. open the jupter notebook service on your port (`19000` here)

## Datasets
### `real2paint` Dataset
For the real domain, please download the [UHDM dataset](https://xinyu-andy.github.io/uhdm-page/) from its official website. For the painting domain, we have curated a dataset of high-resolution Vincent van Gogh paintings, which can be downloaded at [link1](https://github.com/Kaminyou/UHR-Vincent-van-Gogh-real2paint) or [link2](https://www.dropbox.com/scl/fi/gohkhvipij61w496eeqdw/vincent_van_gogh.zip?rlkey=vco57kdadendwhy4zzednkk4i&st=d127g9bk&dl=0). Please note that we do not own these images; users should ensure their use does not trigger legal issues.

### `Kyoto-summer2autumn` Dataset
Please download it at [link](https://github.com/Kaminyou/Kyoto-summer2autumn).

### `ANHIR` Dataset
Please download it at [link](https://anhir.grand-challenge.org/Data/). Please note that we do not own these images; users should ensure their use does not trigger legal issues.

### `ACROBAT` Dataset
Please download it at [link](https://acrobat.grand-challenge.org/). Please note that we do not own these images; users should ensure their use does not trigger legal issues.

## Train your model
The training of I2I model is the same as [KIN](https://github.com/Kaminyou/URUST). DN is a plugin for any I2I model with InstanceNorm layers.

## Citation
```
@InProceedings{10.1007/978-3-031-72995-9_18,
author="Ho, Ming-Yang and Wu, Che-Ming and Wu, Min-Sheng and Tseng, Yufeng Jane",
title="Every Pixel Has Its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="312--328",
isbn="978-3-031-72995-9"
}
```
