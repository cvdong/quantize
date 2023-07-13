# -*- encoding: utf-8 -*-

'''
Version: v1.0
Author: 東
Date: 2023.7.13
Description: onnx-->engine 通过onnx解析生成engine model
             test env: onnx 1.12.1 tensorrt 8.4.1.5 
Copyright (c) 2023 by ${東}, All Rights Reserved. 

'''

import argparse
import logging
from pathlib import Path
from sympy import false, true
import tensorrt as trt
import os
import glob
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torchvision.transforms as transforms
from PIL import Image


Parser = argparse.ArgumentParser()
Parser.add_argument('--onnx_path', type=str, default='weights/yolov8s.onnx')
Parser.add_argument('--has_half', type=bool, default=false)
Parser.add_argument('--has_int8', type=bool, default=false)
Parser.add_argument('--calib_path', type=str, default='data/calib_images')


args = Parser.parse_args()

# logging 接口
logger = trt.Logger(trt.Logger.INFO) # trt.Logger.WARNING

# 日志设置
def set_logging(name=None):
    
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    return logging.getLogger(name)

LOGGER = set_logging('export_engine')

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  
    colors = {'black': '\033[30m',  
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m', 
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

# 返回文件大小
def file_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6
    else:
        return 0.0


class EntropyCalibrator(trt.IInt8MinMaxCalibrator):
    
    def __init__(self, files_path=r'imgs'):
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = 'data/cache.cache'

        self.batch_size = 1
        self.Channel = 3
        self.Height = 640
        self.Width = 640
        self.transform = transforms.Compose([
            transforms.Resize([self.Height, self.Width]),  # [h,w]
            transforms.ToTensor(),
        ])

        self.imgs = glob.glob(os.path.join(files_path, '*'))
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs)//self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel,self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size:\
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = Image.open(f).convert('RGB')
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size/self.batch_size), 'not valid img!'+f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# onnx --> engine
def export_engine(onnx_path, calib, half=False, trt_int8=False, workspace=2, prefix=colorstr('TensorRT:')):
     
    file = Path(onnx_path)
    f = file.with_suffix('.engine')
    
    # basic_set
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30 # workspace * 1G
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    # 解析onnx
    parser = trt.OnnxParser(network, logger)
    
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_path}')
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)] 
    outputs = [network.get_input(i) for i in range(network.num_outputs)] 
    
    # 日志输出
    LOGGER.info(f'{prefix} Network Description:')
    for inp in inputs:
        LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
        
    LOGGER.info(f'{prefix} building FP{16 if half else 32} engine in {f}')

    half &= builder.platform_has_fast_fp16
    
    trt_int8 &= builder.platform_has_fast_fp16
    
    # 量化到FP16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    
    if trt_int8:    
        config.set_flag(trt.BuilderFlag.INT8)
        
        calibrator = EntropyCalibrator(files_path=calib)
        
        config.int8_calibrator = calibrator
        
    # 序列化存储
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
        
    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        
    return f
    

if __name__ == '__main__':
    
    '''
    engine生成可以直接利用shell,也可以使用Tools-->trtexec:
    
    trtexec --onnx = ./weights/yolov5s.onnx  \
            --saveEngine = ./weights/yolov5s_fp16.engine \
            --workspace = 4096
            --fp16
    
    '''
    
    # onnx-->engine
    # Logger-->Builder-->BuilderConfig-->NetWork-->SeralizedNetWork
    export_engine(args.onnx_path, args.calib_path, half=args.has_half, trt_int8=args.has_int8)
    
    print("--咕噜噜----onnx 转换 engine 完成!!!----------")