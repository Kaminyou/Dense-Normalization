FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="Ming-Yang Ho"

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
	DEBIAN_FRONTEND=noninteractive \
	LC_ALL=en_US.UTF-8

RUN apt-get update && \
	apt-get install -y \
	python3.9-dev \
	python3-pip \
	ipython3 \
	git \
	cmake \
	vim \
    locales \
    python3-opencv && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN echo deb http://tw.archive.ubuntu.com/ubuntu/ focal main restricted >> /etc/apt/sources.list && \
	apt-key adv --no-tty --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3B4FE6ACC0B21F32 && \
	apt update && \
	rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && \
	update-locale && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 10 && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
	update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10 && \
	update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
	ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10 && \
	pip --no-cache-dir install --upgrade pip

ARG TORCH_VER="1.13.0" \
    TORCH_VISION_VER="0.14.0"

ARG CUDA_VER="cu116"

RUN pip --no-cache-dir install \
    torch==${TORCH_VER}+${CUDA_VER} \
    torchvision==${TORCH_VISION_VER}+${CUDA_VER} \
    -f "https://download.pytorch.org/whl/${CUDA_VER}/torch_stable.html"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
RUN pip install jupyter

CMD ["/bin/bash"]