# https://blog.softwaremill.com/setting-up-tensorflow-with-gpu-acceleration-the-quick-way-add80cd5c988
# docker build -t <your image name>/tensorflow-gpu-jupyter .
# docker run -it --rm --gpus all -p 8888:888 tensorflow/tensorflow:latest-gpu-jupyter
FROM tensorflow/tensorflow:latest-gpu-jupyter
# ^or just latest-gpu if you don't need Jupyter

WORKDIR /NOCS

# Set desired Python version
#ENV python_version 3

# Install desired Python version (the current TF image is be based on Ubuntu at the moment)
#RUN apt install -y python${python_version}

# Set default version for root user - modified version of this solution: https://jcutrer.com/linux/upgrade-python37-ubuntu1810
#RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python${python_version} 1

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN python -m pip install --upgrade pip setuptools wheel
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN pip install ninja opencv-python numpy pillow pycocotools matplotlib tqdm scikit-image keras==2.2.0

# By copying over requirements first, we make sure that Docker will "cache"
# our installed requirements in a dedicated FS layer rather than reinstall
# them on every build
#COPY requirements.txt requirements.txt

# Install the requirements
#RUN python -m pip install -r requirements.txt

# Only needed for Jupyter
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "jupyter notebook --notebook-dir=/NOCS --ip 0.0.0.0 --no-browser --allow-root"]
