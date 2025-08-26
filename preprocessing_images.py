import os
import numpy as np

import torch
import torchvision.transforms as transforms

import cv2
from PIL import Image
from preprocess_pipeline import RemoveSpecularHighlights, LightingNormalization

MAIN_DATASET_PATH = './datasets/'
FOLDER = 'sessile-main-Kvasir-SEG'

def get_files_from_folder(FOLDER):
    folder_path = os.path.join(MAIN_DATASET_PATH, FOLDER)
    images_path = os.path.join(folder_path, 'images')
    masks_path = os.path.join(folder_path, 'masks')
    list_images = os.listdir(images_path)
    list_masks = os.listdir(masks_path)
    return list_images,list_masks

def adjust_image(path):
    im = Image.open(path)
    img_tensor = transforms.functional.to_tensor(im)
    model = RemoveSpecularHighlights()
    output_img = model(img_tensor)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    return output_img

##############################################################################################################################


images,masks = get_files_from_folder(FOLDER)

for i,m in zip(images,masks):
    path_file_image = str(MAIN_DATASET_PATH + '/' + FOLDER + '/' + 'images' + '/' + i)
    path_file_mask = str(MAIN_DATASET_PATH + '/' + FOLDER + '/' + 'masks' + '/' + m)

    pretrained_image = adjust_image(path_file_image)
    mask = cv2.imread(path_file_mask)

    path_file_image_after = str(MAIN_DATASET_PATH + 'preprocessed' + '/' + FOLDER + '/' + 'images' + '/' + i)
    path_file_mask_after = str(MAIN_DATASET_PATH + 'preprocessed' + '/' + FOLDER + '/' + 'masks' + '/' + m)
    
    os.makedirs(os.path.dirname(path_file_image_after), exist_ok=True)
    os.makedirs(os.path.dirname(path_file_mask_after), exist_ok=True)
    
    cv2.imwrite(path_file_image_after, pretrained_image)
    cv2.imwrite(path_file_mask_after, mask)


##############################################################################################################################
