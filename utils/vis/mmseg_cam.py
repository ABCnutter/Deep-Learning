##################################################################################################################################################################
 
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from PIL import Image
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np 
from collections import OrderedDict
from typing import Dict, Iterable, Callable
from torch import nn, Tensor
from pprint import pprint
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
import json
import argparse
from dataclasses import dataclass

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from mmseg.apis import init_model, inference_model


# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}
# 1_Crack_105_79_19.jpg   1_Crack_636_88_176.jpg  1_Crack_812_90_148.jpg
# 1_Crack_322_81_227.jpg  1_Crack_792_89_308.jpg
# 1_Crack_481_87_26.jpg   1_Crack_796_89_333.jpg
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
image_file = "/home/lpx/NewDisk2/chenpeng/seg_dev_sec_laster/seg_dev_sec/code/dataset/crackdataset_new/cam"
image_name = "1_Crack_481_87_26"
IMAGE_FILE_PATH = os.path.join(image_file, image_name + (".jpg"))
MEAN = [0.5835, 0.5820, 0.5841]
STD = [0.1149, 0.1111, 0.1064]
    
CONFIG = 'work_dir/swin-tiny-patch4-window7_upernet/swin-tiny-patch4-window7_upernet_1xb8-20k_levir-256x256.py'
CHECKPOINT = 'work_dir/swin-tiny-patch4-window7_upernet/iter_25600.pth'
PREVIEW_MODEL = True 
# TARGET_LAYERS = ["model.model.backbone.layer4"] # TARGET_LAYERS请在main函数中修改,已标注修改位置
METHOD =  'GradCAM'
SEM_CLASSES = ['crack']
TARGET_CATEGORY = 'crack'
VIS_CAM_RESULTS = True
CAM_SAVE_PATH = "/home/lpx/NewDisk2/chenpeng/openmmlab/mmsegmentation/work_dir/cam"
LIKE_VIT = True
PRITN_MODEL_PRED_SEG = False


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img', default=IMAGE_FILE_PATH, help='Image file')
    parser.add_argument('--config', default=CONFIG ,help='Config file')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Checkpoint file')
    # parser.add_argument(
    #     '--target_layers',
    #     default=TARGET_LAYERS,
    #     nargs='+',
    #     type=str,
    #     help='The target layers to get CAM, if not set, the tool will '
    #     'specify the norm layer in the last block. Backbones '
    #     'implemented by users are recommended to manually specify'
    #     ' target layers in commmad statement.')
    parser.add_argument(
        '--preview_model',
        default=PREVIEW_MODEL,
        help='To preview all the model layers')
    
    parser.add_argument(
        '--method',
        default=METHOD,
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    
    parser.add_argument(
        '--sem_classes',
        default=SEM_CLASSES,
        nargs='+',
        type=int,
        help='all classes that model predict.')
    parser.add_argument(
        '--target_category',
        default=TARGET_CATEGORY,
        type=str,
        help='The target category to get CAM, default to use result '
        'get from given model.')

    parser.add_argument(
        '--aug_mean',
        default=MEAN,
        nargs='+',
        type=float,
        help='augmentation mean')
    
    parser.add_argument(
        '--aug_std',
        default=STD,
        nargs='+',
        type=float,
        help='augmentation std')
    
    parser.add_argument(
        '--cam_save_path',
        default=CAM_SAVE_PATH,
        type=str,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument(
        '--vis_cam_results',
        default=VIS_CAM_RESULTS)
    parser.add_argument('--device', default=DEVICE, help='Device to use cpu')
    
    parser.add_argument(
        '--like_vision_transformer',
        default=LIKE_VIT,
        help='Whether the target model is a ViT-like network.')
    
    parser.add_argument(
        '--print_model_pred_seg',
        default=PRITN_MODEL_PRED_SEG,
        help='')

    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args




def make_input_tensor(image_file_path, mean, std,  device):
    if not os.path.exists(image_file_path):
        raise(f"{image_file_path} is not exist!")
    img = Image.open(image_file_path)
    img_array = np.array(img)
    rgb_img = np.float32(img_array) / 255      
    input_tensor = preprocess_image(rgb_img, mean=mean, std=std)
    if device == torch.device('cuda:0'):
        input_tensor = input_tensor.to(device)
    print(f"input_tensor has been to {device}")
    return input_tensor, rgb_img
    

def make_model(config_path, checkpoint_path, device):
    # 从配置文件和权重文件构建模型
    model = init_model(config_path, checkpoint_path, device=device)
    print('网络设置完毕 ：成功载入了训练完毕的权重。')
    return model


from torch.nn import functional as F
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = F.interpolate(self.model(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

def reshape_transform_fc(in_tensor):
    result = in_tensor.reshape(in_tensor.size(0),
        int(np.sqrt(in_tensor.size(1))), int(np.sqrt(in_tensor.size(1))), in_tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result



def main():
    args = parse_args()
    
    input_tensor, rgb_img = make_input_tensor(args.img, args.aug_mean, args.aug_std, device=args.device)
    
    cfg = args.config
    checkpoint = args.checkpoint
    model_mmseg = make_model(cfg, checkpoint, device=args.device)
    
    results= inference_model(model_mmseg, args.img)
    
    if args.print_model_pred_seg:
        # 推理给定图像
        pprint(results)

    if args.preview_model:
        print('模型modules如下:')
        pprint([name for name, _ in model_mmseg.named_modules()])
    
    model = SegmentationModelOutputWrapper(model_mmseg)
    output = model(input_tensor)

    sem_classes = args.sem_classes
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    if len(sem_classes) == 1:
        output = torch.nn.functional.sigmoid(output).cpu()
        perd_mask = torch.where(output > 0.3, torch.ones_like(output), torch.zeros_like(output))
        perd_mask = perd_mask.detach().cpu().numpy()
        
    else:
        output = torch.nn.functional.softmax(output, dim=1).cpu()
        perd_mask = output[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    
    category = sem_class_to_idx[args.target_category]
    mask_float = np.float32(perd_mask == category)

    # reshape_transform = reshape_transform_fc if args.like_vision_transformer else None
    
    ##########################################################################################################################################################################
    
    target_layers = [model.model.backbone.norm3]

    ##########################################################################################################################################################################
    targets = [SemanticSegmentationTarget(category, mask_float)]
    GradCAM_Class = METHOD_MAP[args.method.lower()]
    with GradCAM_Class(model=model,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available(),
                reshape_transform=reshape_transform_fc if args.like_vision_transformer else None
                ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    vir_image = Image.fromarray(cam_image)
    
    if args.vis_cam_results:
        vir_image.show()
    cam_save_path = f"{args.cam_save_path}/{os.path.basename(args.config).split('.')[0]}"
    if not os.path.exists(cam_save_path):
        os.makedirs(cam_save_path)
    vir_image.save(os.path.join(cam_save_path, f"{os.path.basename(args.img).split('.')[0]}.png"))

if __name__ == '__main__':
    
    main()
