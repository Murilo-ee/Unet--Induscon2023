import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


#######################################################
FOLDER = 'TEST_IMAGES'
FILES = ['133_cvc.jpg','419_cvc.jpg','480_cvc.jpg','cju1cfhyg48bb0799cl5pr2jh_kev.jpg','cju2hqt33lmra0988fr5ijv8j_kev.jpg']
MODEL_NAME = 'UNET_with_preprocess_no_crop_v5.h5'
#######################################################

model = tf.keras.models.load_model('./models/{}'.format(MODEL_NAME),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})

paths = []
paths_mask = []

for file in FILES:
    path_img = './datasets/{}/images/{}'.format(FOLDER,file)
    path_mask = './datasets/{}/masks/{}'.format(FOLDER,file)

    paths.append(path_img)
    paths_mask.append(path_mask)

array = get_files(paths,type_of_file='image')
array_mask = get_files(paths_mask,type_of_file='mask')

prediction = model.predict(array)
binario = np.where(prediction > 0.5, 1, 0)

arrays_mask = tf.cast(array_mask, tf.int32)
arrays_predict = tf.cast(binario, tf.int32)

if len(arrays_mask.shape) < 4:
    arrays_mask = tf.expand_dims(arrays_mask, axis=-1)  # [batch_size, height, width, 1]
    arrays_mask = tf.repeat(arrays_mask, repeats=1, axis=-1)  # [batch_size, height, width, channels]

print(arrays_mask.shape)
print(arrays_predict.shape)

ssim = score_ssim(arrays_predict,arrays_mask)
print('SSIM:', ssim)
mse = score_mse(arrays_predict,arrays_mask)
print('MSE:', mse)
jaccard = score_jaccard(arrays_predict,arrays_mask)
print('JACCARD:', jaccard)
euclidean = score_euclidean(arrays_predict,arrays_mask)
print('EUCLIDEAN:', euclidean)
precision,recall = confusion_matrix(arrays_predict,arrays_mask)
print('PRECISION:', precision)
print('RECALL:', recall)
