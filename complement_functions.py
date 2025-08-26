import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend import epsilon
import matplotlib.pyplot as plt
import re

def score_ssim(arrays_predict,arrays_mask):
    ssim = tf.image.ssim(arrays_mask, arrays_predict, max_val=1.0)
    return ssim.numpy().mean()

def score_mse(arrays_predict,arrays_mask):
    mse = np.mean((arrays_predict - arrays_mask) ** 2)
    return mse

def score_jaccard(arrays_predict,arrays_mask):
    intersection = np.logical_and(arrays_predict,arrays_mask).sum()
    union = np.logical_or(arrays_predict,arrays_mask).sum()
    jaccard_index = intersection / union
    return jaccard_index

def score_euclidean(arrays_predict,arrays_mask):
    euclidean_distance = np.sqrt(np.sum((arrays_predict - arrays_mask) ** 2))
    return euclidean_distance

def confusion_matrix(arrays_predict,arrays_mask):
    true_positive = np.logical_and(arrays_mask, arrays_predict).sum()
    false_positive = np.logical_and(1 - arrays_mask, arrays_predict).sum()
    false_negative = np.logical_and(arrays_mask, 1 - arrays_predict).sum()
    true_negative = np.logical_and(1 - arrays_mask, 1 - arrays_predict).sum()
    # out of all the positive predicted, what percentage is truly positive
    precision = true_positive / (true_positive + false_positive)
    # out of the total positive, what percentage are predicted positive
    recall = true_positive / (true_positive + false_negative)
    return precision,recall

def get_files(paths,type_of_file='image'):
    all_images = []
    for path in paths:
        if type_of_file == 'image':
            img = cv2.imread(path)
            img = img.astype('float32') / 255.0
            
        elif type_of_file == 'mask':
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('float32') / 255.0
            _, img = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            
        all_images.append(img)
    arrays = np.array(all_images)
    return arrays

def image_mask_generator(image_paths, mask_paths, batch_size):
    num_samples = len(image_paths)
    while True:
        indices = np.random.choice(num_samples, batch_size)
        batch_images = []
        batch_masks = []
        for i in indices:
            img = cv2.imread(image_paths[i])
            img = img.astype('float32') / 255.0
            img = np.array(img)
            # image = np.array(Image.open(image_paths[i]))

            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('float32') / 255.0
            _, mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
            mask = np.array(mask)
            # mask = np.array(Image.open(mask_paths[i]))
            
            batch_images.append(img)
            batch_masks.append(mask)
        yield np.array(batch_images), np.array(batch_masks)

def get_folders(list_folders,test_size):

    images_full_list = []
    masks_full_list = []

    for folder in list_folders:
        main_path = str('./datasets/' + folder)
        images_path = str(main_path + '/images/')
        masks_path = str(main_path + '/masks/')
        images_path_sub = os.listdir(images_path)
        masks_path_sub = os.listdir(masks_path)

        images_path_complete = [images_path+x for x in images_path_sub]
        masks_path_complete = [masks_path+x for x in masks_path_sub]

        images_full_list.extend(images_path_complete)
        masks_full_list.extend(masks_path_complete)
    
    images_train, images_test, masks_train, masks_test = train_test_split(
        images_full_list,masks_full_list,test_size=test_size,random_state=42)

    return images_train, images_test, masks_train, masks_test

def count_pixel_intensity(array):
    # Transforma o array em um array unidimensional
    flat_array = array.ravel()
    
    # Cria um vetor de 256 zeros para armazenar a contagem de pixels de cada intensidade
    count = np.zeros(256, dtype=np.int32)
    
    # Percorre o array unidimensional e conta a quantidade de pixels de cada intensidade
    for i in range(256):
        count[i] = np.sum(flat_array == i)
    
    return count

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * (1 - p_t) ** gamma * bce(y_true, y_pred)
        return loss
    return loss


def dice_loss(y_true, y_pred):
    """
    Define o Dice Loss para o treinamento da U-Net.
    
    Args:
        y_true (array): Máscara binária do ground truth.
        y_pred (array): Máscara binária prevista pelo modelo.
    
    Returns:
        float: Valor do Dice Loss.
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / (denominator + epsilon())


class Threshold(tf.keras.layers.Layer):
    def __init__(self, threshold, **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.threshold = threshold
    
    def call(self, inputs):
        return tf.math.round(inputs - self.threshold + 0.5)

def apply_threshold(x):
    return tf.where(x > 0.5, 1.0, 0.0)



class ThresholdLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ThresholdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                      trainable=True)
        super(ThresholdLayer, self).build(input_shape)

    def call(self, x):
        return tf.keras.backend.sigmoid(100*(x-self.kernel))

    def compute_output_shape(self, input_shape):
        return input_shape

class GetHistory():
    def __init__(self, df):
        self.df = df
    
    def accuracy_vs_val_accuracy(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.df.index, self.df['accuracy'], label='Train')
        ax.plot(self.df.index, self.df['val_accuracy'], label='Validation')
        ax.set_title('Train and Validation Accuracy',
                     fontdict={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_xlabel('Epochs', fontdict={'fontsize': 12})
        ax.set_ylabel('Accuracy', fontdict={'fontsize': 12})
        colors = ['#1a3cad', '#ff0e26']
        for i, line in enumerate(ax.get_lines()):
            line.set_color(colors[i])
            line.set_linewidth(2.5)
        ax.legend()
        plt.show()
        return fig
        
    def loss_vs_val_loss(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.df.index, self.df['loss'], label='Train')
        ax.plot(self.df.index, self.df['val_loss'], label='Validation')
        ax.set_title('Train and Validation Loss',
                     fontdict={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_xlabel('Epochs', fontdict={'fontsize': 12})
        ax.set_ylabel('Loss', fontdict={'fontsize': 12})
        colors = ['#1a3cad', '#ff0e26']
        for i, line in enumerate(ax.get_lines()):
            line.set_color(colors[i])
            line.set_linewidth(2.5)
        ax.legend()
        plt.show()
        return fig