FROM debian:12

RUN apt-get update &&  DEBIAN_FRONTEND="noninteractive" TZ="Europe/Zurich" apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    cmake wget vim htop \
    python3 python3-dev python3-pip python-is-python3 \
    hdf5-tools h5utils \
    zip \
    unzip \
    ssh \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
    torch==2.1.0 \
    torchvision==0.16 \
    torchmetrics==1.3.2 \
    numpy==1.25.2 \
    tqdm==4.66.2 \
    scipy==1.11.4 \
    scikit-image==0.20.0 \
    scikit-learn==1.3.2 \
    seaborn==0.13.0 \
    matplotlib==3.8.1 \
    tensorboard==2.15.2 \
    pydicom==2.4.4 \
    pillow imageio h5py \
	pandas \
	opencv-contrib-python-headless \
    jupyter \
    jupyterlab \
    ipywidgets \
    pyxu \

RUN jupyter nbextension enable --py widgetsnbextension

USER root
RUN mkdir /opt/lab
COPY setup.sh /opt/lab/
COPY local_files /opt/lab/
RUN chmod -R a+x /opt/lab/