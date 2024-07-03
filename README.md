# [ECCV 2024] Dense-Normalization
Every Pixel Has its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization

## Get Started with an example
We provide a simple example (one image from the Kyoto summer2autumn dataset) for you to translate an UHR image with our DN

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
For the real domain, please download the [UHDM dataset](https://xinyu-andy.github.io/uhdm-page/) from its official website. For the painting domain, we have curated a dataset of high-resolution Vincent van Gogh paintings, which can be downloaded at [link](https://www.dropbox.com/scl/fi/gohkhvipij61w496eeqdw/vincent_van_gogh.zip?rlkey=vco57kdadendwhy4zzednkk4i&st=d127g9bk&dl=0). Please note that we do not own these images; users should ensure their use does not trigger legal issues.

### `Kyoto-summer2autumn` Dataset
Please download it at [link](https://github.com/Kaminyou/Kyoto-summer2autumn).

### `ANHIR` Dataset
Please download it at [link](https://anhir.grand-challenge.org/Data/). Please note that we do not own these images; users should ensure their use does not trigger legal issues.

## Train your model
The training of I2I model is the same as [KIN](https://github.com/Kaminyou/URUST). DN is a plugin for any I2I model with InstanceNorm layers.
