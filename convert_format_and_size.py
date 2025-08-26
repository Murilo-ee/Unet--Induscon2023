import os
import cv2
import pandas as pd
import tifffile

IN_FORMAT = 'tif'
OUT_FORMAT = 'jpg'

OUT_SIZE_X = 256
OUT_SIZE_Y = 256

MAIN_DATASET_PATH = './datasets/'
FOLDER = 'CVC-ClinicDB'

def get_files_from_folder(FOLDER):
    folder_path = os.path.join(MAIN_DATASET_PATH, FOLDER)
    images_path = os.path.join(folder_path, 'images')
    masks_path = os.path.join(folder_path, 'masks')
    list_images = os.listdir(images_path)
    list_masks = os.listdir(masks_path)
    return list_images,list_masks

def convert_type(IN_FORMAT, OUT_FORMAT, path_file):
    if IN_FORMAT=='tif':

        imagem_tif = tifffile.imread(path_file)
        new_path_file = str('.'.join(path_file.split('.')[:-1])+'.'+OUT_FORMAT)
        tifffile.imwrite(new_path_file, imagem_tif)

        os.remove(path_file)

    else:

        new_path_file = path_file

    print(new_path_file)
    return new_path_file

def convert_size(path_file,x,y):
    file = cv2.imread(path_file)
    # print(file.shape)
    file = cv2.resize(file,(y,x))
    cv2.imwrite(path_file, file)
    # print(file.shape)



##############################################################################################################################


images,masks = get_files_from_folder(FOLDER)

for i,m in zip(images,masks):
    path_file_image = str(MAIN_DATASET_PATH + '/' + FOLDER + '/' + 'images' + '/' + i)
    path_file_mask = str(MAIN_DATASET_PATH + '/' + FOLDER + '/' + 'masks' + '/' + m)

    path_file_image_type = convert_type(IN_FORMAT, OUT_FORMAT, path_file_image)
    path_file_mask_type = convert_type(IN_FORMAT, OUT_FORMAT, path_file_mask)

    convert_size(path_file_image_type,OUT_SIZE_X,OUT_SIZE_Y)
    convert_size(path_file_mask_type,OUT_SIZE_X,OUT_SIZE_Y)


##############################################################################################################################

# TESTS

# new_path_file = convert_type(IN_FORMAT, OUT_FORMAT, './datasets/CVC-ClinicDB/images/1.tif')
# convert_size(new_path_file,OUT_SIZE_X,OUT_SIZE_Y)
