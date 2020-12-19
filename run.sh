#!/bin/bash

SOURCE="/home/zhangniansong/proxylessnas/deploy/caffe_nets"
RESULT="/home/zhangniansong/proxylessnas/deploy/compiled"
GPU=0

python compile.py \
    -s $SOURCE \
    -r $RESULT \
    -g $GPU