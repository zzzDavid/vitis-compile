# Vitis-AI-Caffe Batch Compilation

## Introduction

This tool is for batch quantization and compilation of caffe models in vitis docker container.

## Usage

This tool needs to run in vitis docker container. It calls `vai_q_caffe` and `vai_c_caffe` to quantize and compile target model.


To start a vitis docker container（for example）：
```bash
$ docker run -it --runtime=nvidia --name=vitis -p 8012:8012 -v /home/zhangniansong/:/home/zhangniansong/ 192.168.3.224:8083/compiler_dnnk_rknn/vitis-ai-tools-1.0.0-gpu /bin/bash
```
Note：
- `--runtime=nvidia` use nvidia container runtime (to use GPU)
- `-p` specifies the port to use
- `-v` mount specified directory as volume to the container, the format is: `/path/in/host:/path/in/container`
- `192.168.3.224:8083/compiler_dnnk_rknn/vitis-ai-tools-1.0.0-gpu`： The name of the vitis docker image

Once enter the docker container, cd to this tool's directory and run:
```sh
$ conda activate vitis-ai-caffe
$ ./run.sh
```
The scipt calls `compile.py` with a number of arguments:
- `-s`: the source folder, contains all the `*.prototxt` and `*.caffemodel` files to be quantized and compiled.
- `-r`: the result folder, the compiled `*.elf` files will be saved here.
- `-g`: the gpu to use.


