# Azure ML


## Running with Docker

### Quick Start

- loging to registry `docker login -u [USERNAME] -p [PASWORD] [USERNAME].azurecr.io`
- pull docker `docker pull [USERNAME].azurecr.io/yolact:1`
- run `docker run --gpus=all --shm-size 8G -v /home/$USER/mnt/project_zero/:/mnt/ -it yolacter`


__Key Requirement:__ [NVIDIA Driver Installation](https://github.com/NVIDIA/nvidia-docker) 

#### Usage

__Train__

TODO:

__Evaluate__

TODO:


__Prediction/Inference__

TODO:

## Build Container

```
docker login 
# from ../research/AzureML/
docker build . -f Docker/yolact.Dockerfile -t [USERNAME].azurecr.io/yolact:1
docker push [USERNAME].azurecr.io/yolact:1
```