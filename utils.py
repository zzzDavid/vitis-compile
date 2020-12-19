"""
This tool is to remove redundant ImageData layer after Vitis quantization
This is a bug of Vitis caffe quantization tool. It cannot correctly replace the ImageData layer
with input layer.
"""
from Caffe import caffe_net
import os

def remove_ImageDataLayer(protoxt_path):
    if not os.path.exists(protoxt_path):
        raise Exception("prototxt file does not exist")
    cnet = caffe_net.Prototxt(protoxt_path)
    input_count = len([layer for layer in cnet.layers() if layer.type == "Input"])
    for i, layer in enumerate(cnet.layers()):
        if layer.type == "Input": 
            if input_count == 2:
                del cnet.net.layer[i]
                break
        if layer.type == "ImageData":            
            del cnet.net.layer[i]
            break
    os.remove(protoxt_path)
    cnet.save_prototxt(protoxt_path)        
