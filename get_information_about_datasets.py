import os
import cv2
import pandas as pd

MAIN_DATASET_PATH = './datasets/'
SELECTION_FILE_NUMBER = 18

def search_folders():
    folders_inside_datasets = os.listdir(MAIN_DATASET_PATH)
    print("Quantidade de pastas detectadas:", len(folders_inside_datasets))
    return folders_inside_datasets

def explore_files(file, type):
    img = cv2.imread(file)
    print("O tamanho da {} é".format(type),img.shape)
    print("O tipo da {} é".format(type),img.dtype)
    return img

def explore_folders(folder,comparative_table):
    folder_path = os.path.join(MAIN_DATASET_PATH, folder)
    images_path = os.path.join(folder_path, 'images')
    masks_path = os.path.join(folder_path, 'masks')

    images = os.listdir(images_path)
    masks = os.listdir(masks_path)
    
    print("ARQUIVO:", folder, "\n")
    print("NUMERO DE IMAGENS:", len(images))
    print("NUMERO DE MASCARAS:", len(masks))

    image_path = os.path.join(images_path, images[SELECTION_FILE_NUMBER])
    mask_path = os.path.join(masks_path, masks[SELECTION_FILE_NUMBER])

    format_img = images[SELECTION_FILE_NUMBER].split('.')[-1]
    format_mask = masks[SELECTION_FILE_NUMBER].split('.')[-1]

    image = explore_files(image_path, 'imagem')
    mask = explore_files(mask_path, 'mascara')

    comparative_table = comparative_table.append({'folder':folder,'format':'{}/{}'.format(format_img,format_mask),
                                                  'num_images':len(images),'num_masks':len(masks),'shape_images':image.shape,
                                                  'shape_masks':mask.shape},ignore_index=True)

    return comparative_table


##############################################################################################################################

comparative_table = pd.DataFrame(columns=['folder','format','num_images','num_masks','shape_images','shape_masks'])

folders = search_folders()
for folder in folders:
    comparative_table = explore_folders(folder,comparative_table)
    print("\n")

print(comparative_table)

##############################################################################################################################
