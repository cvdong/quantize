'''
Version: v1.0
Author: 東DONG
Mail: cv_yang@126.com
Date: 2022-11-14 15:15:36
LastEditTime: 2022-11-17 16:48:44
FilePath: /Inference/ppq/export_onnx.py
Description: 

Copyright (c) 2022 by ${東}, All Rights Reserved. 

                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='

     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑     永不宕机     永无BUG

'''

import os
import torchvision.transforms as transforms
from PIL import Image
from ppq import *
from ppq.api import *

ONNX_PATH        = 'modified_best.onnx'   
ENGINE_PATH      = 'Quantized.onnx'  
JSON_PATH        = 'Quantized.json'
CALIBRATION_PATH = '/datasets/VOC/images/val2007'        
BATCHSIZE        = 1
EXECUTING_DEVICE = 'cuda'

# dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([640, 640]),
    transforms.ToTensor(),
])

for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img)
        
    print(f'-----------------------runing:  {file}----------------')

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)



# ppq
with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        platform=TargetPlatform.TRT_INT8,
        onnx_import_file=ONNX_PATH, 
        calib_dataloader=dataloader, 
        calib_steps=32, device=EXECUTING_DEVICE,
        input_shape=[BATCHSIZE, 3, 640, 640], 
        collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    snr_report = graphwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    snr_report = layerwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    
    export_ppq_graph(
        qir, platform=TargetPlatform.TRT_INT8, 
        graph_save_to=ENGINE_PATH, 
        config_save_to=JSON_PATH)