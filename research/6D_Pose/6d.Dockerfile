FROM tensorflow/tensorflow:1.12.0-gpu-py3

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


RUN pip install ninja opencv-python pillow pycocotools matplotlib tqdm scikit-image keras==2.2.0