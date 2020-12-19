import json
import os
import subprocess
import argparse
import settings
from glob import glob
from utils import remove_ImageDataLayer


parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--source',
    help="source directory of caffe model files (*.prototxt, *.caffemodel)",
    type=str,
    default='./source'
)
parser.add_argument(
    '-r',
    '--result',
    help="result directory of output files (*.elf)",
    type=str,
    default='./results'
)
parser.add_argument(
    '-g',
    '--gpu',
    help="which gpu to use",
    type=int,
    default=0
)
parser.add_argument(
    '-m',
    '--mode',
    help='compilation mode',
    type=str,
    default='debug'
)
parser.add_argument(
    '-a',
    '--arch',
    help='DPU architecture',
    type=str,
    default='ZCU102'
)

args = parser.parse_args()

model_type = "caffe"
result_dir = args.result
gpu_index = args.gpu
source_dir = args.source


for item in glob(os.path.join(source_dir, '*.prototxt')):

    item = os.path.basename(item)
    item = item.split(".")[0]

    tmp_path = os.path.join(source_dir, item)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    
    elf_path = os.path.join(result_dir, "dpu_" + item + ".elf")
    if os.path.exists(elf_path): 
        print(elf_path + " already exists")
    else:
        caffe_env = settings.cmd_caffe_env
        caffe_decent = settings.cmd_caffe_decent.format(source_dir, result_dir, item, gpu_index)
        print(caffe_decent)

        subprocess.run(caffe_decent, shell=True)

        deploy_prototxt = os.path.join(source_dir, item, 'quantize_results', 'deploy.prototxt')
        # a bug of Vitis: there's a duplicate input layer in the quantized deploy.prototxt
        remove_ImageDataLayer(deploy_prototxt)

        caffe_dnnc = settings.cmd_caffe_dnnc.format(source_dir,
                                                    result_dir,
                                                    item,
                                                    gpu_index,
                                                    settings.ARCH[args.arch],
                                                    "{'mode': '" + args.mode + "'}")
        print(caffe_dnnc)                                            
        subprocess.run(caffe_dnnc, shell=True)

