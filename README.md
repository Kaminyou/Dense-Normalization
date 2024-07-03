# [ECCV 2024] Dense-Normalization
Every Pixel Has its Moments: Ultra-High-Resolution Unpaired Image-to-Image Translation via Dense Normalization

## Environment preparation
1. Please check your GPU driver version and modify `Dockerifle` accordingly
2. Then, execute
    ```
    $ docker-compose up --build -d
    ```
3. Get into the docker container
    ```
    $ docker exec -it dn-env bash
    ```

## Inference
In the docker container, please execute
```
$ python3 transfer.py -c data/real_to_watercolor/config.yaml --skip_cropping
```