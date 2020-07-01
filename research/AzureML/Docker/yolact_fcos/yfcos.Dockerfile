# https://github.com/bongjoonhyun/fcos/tree/master/docker
FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER ZeroWaste


ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get clean
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python-pip
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y apt-utils
RUN apt-get install -y curl
RUN apt-get install -y ca-certificates
RUN apt-get install -y bzip2
RUN apt-get install -y cmake
RUN apt-get install -y tree
RUN apt-get install -y htop
RUN apt-get install -y bmon
RUN apt-get install -y iotop
RUN apt-get install -y tmux
RUN apt-get install -y wget
RUN apt-get install -y gdb
RUN apt-get install -y python3-dbg

# Install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH=/root/anaconda3/bin:$PATH

RUN conda create -y --name py36 python=3.6.7

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/root/anaconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja
RUN pip install yacs
RUN pip install cython
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install numpy
RUN pip install azureml-defaults
#azure-storage-blob

# Install PyTorch
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install pycocotools
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


#WORKDIR /ZeroWaste

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git
# set FORCE_CUDA because during `docker build` cuda is not accessible
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

ENV FORCE_CUDA="1"
RUN python -m pip install -e detectron2

# Install Yolact
COPY ./yfcos_post_processing.py /detectron2/detectron2/modeling/postprocessing.py
