import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

class EvaluateModel:
    def __init__(self, FOLDER, FILES, MODEL_NAME,posprocessing):
        self.FOLDER = FOLDER
        self.FILES = FILES
        self.MODEL_NAME = MODEL_NAME
        self.posprocessing = posprocessing

    def load_model(self):
        model = tf.keras.models.load_model('./models/{}.h5'.format(self.MODEL_NAME),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})
        return model
    
    def save_history(self):
        history_name = 'history/{}.csv'.format(self.MODEL_NAME)
        history_table = pd.read_csv(history_name)
        history_class = GetHistory(history_table)
        acc = history_class.accuracy_vs_val_accuracy()
        loss = history_class.loss_vs_val_loss()

        acc.savefig("output_midia/{}_accuracy.png".format(self.MODEL_NAME))
        loss.savefig("output_midia/{}_loss.png".format(self.MODEL_NAME))
        print('HISTORY SAVED')
    
    def get_paths(self):
        paths = []
        paths_mask = [] 
        for FILE in self.FILES:
            path_img = './datasets/{}/images/{}'.format(self.FOLDER,FILE)
            path_mask = './datasets/{}/masks/{}'.format(self.FOLDER,FILE)
            paths.append(path_img)
            paths_mask.append(path_mask)
        
        return paths, paths_mask
    
    def apply_posprocessing(self,mask):
        # mask = (mask * 255).astype(np.uint8)
        mask = (mask).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return filtered

    def save_masks(self):
        paths, paths_mask = self.get_paths()
        model = self.load_model()

        for path in paths:
            array = get_files([path],type_of_file='image')
            prediction = model.predict(array)
            binario = np.where(prediction > 0.5, 1, 0)
            if self.posprocessing == True:
                final_token = '_posprocessed'
                output = self.apply_posprocessing(binario[0])
            elif self.posprocessing == False:
                output = binario[0]
                final_token = ''
            # cv2.imshow(output, interpolation='nearest')
            cv2.imwrite("output_midia/{}_{}{}.png".format(re.search(r'[\\/]([\w-]+)\.jpg', path).group(1),
                                                          self.MODEL_NAME,
                                                          final_token), output*255)
            # plt.imshow(output, interpolation='nearest')
            # plt.savefig("output_midia/{}_{}{}.png".format(re.search(r'[\\/]([\w-]+)\.jpg', path).group(1),
            #                                               self.MODEL_NAME,
            #                                               final_token))

    def get_performance(self):
        ssim = []
        mse = []
        jaccard = []
        euclidean = []
        precision = []
        recall = []

        paths, paths_mask = self.get_paths()
        model = self.load_model()

        for path,path_mask in zip(paths, paths_mask):
            array = get_files([path],type_of_file='image')
            array_mask = get_files([path_mask],type_of_file='mask') 
            prediction = model.predict(array)
            binario = np.where(prediction > 0.5, 1, 0)
            if self.posprocessing == True:
                final_token = '_posprocessed'
                output_pre = self.apply_posprocessing(binario[0])
                output = np.zeros((1, 256, 256, 1), dtype=np.int32)
                output[0, :, :, 0] = output_pre.astype(np.int32)

            elif self.posprocessing == False:
                output = binario
                final_token = ''
            
            arrays_mask = tf.cast(array_mask, tf.int32)
            arrays_predict = tf.cast(output, tf.int32) 
            if len(arrays_mask.shape) < 4:
                arrays_mask = tf.expand_dims(arrays_mask, axis=-1)  # [batch_size, height, width, 1]
                arrays_mask = tf.repeat(arrays_mask, repeats=1, axis=-1)  # [batch_size, height, width, channels]

            ssim.append(score_ssim(arrays_predict,arrays_mask))
            mse.append(score_mse(arrays_predict,arrays_mask))
            jaccard.append(score_jaccard(arrays_predict,arrays_mask))
            euclidean.append(score_euclidean(arrays_predict,arrays_mask))
            prec,rec = confusion_matrix(arrays_predict,arrays_mask)
            precision.append(prec)
            recall.append(rec)

        with open("output_midia/metrics.txt", "a") as f:
            f.write("MODEL: {}{}\n".format(self.MODEL_NAME,final_token))
            f.write("FILES: {}\n".format(self.FILES))
            f.write("ssim: {}\n".format(np.mean(ssim)))
            # f.write("mse: {}\n".format(np.mean(mse)))
            f.write("jaccard: {}\n".format(np.mean(jaccard)))
            f.write("euclidean: {}\n".format(np.mean(euclidean)))
            f.write("precision: {}\n".format(np.mean(precision)))
            f.write("recall: {}\n\n".format(np.mean(recall)))

if __name__ == "__main__":
    print('inicio')
    FOLDER = 'preprocessed/TEST_IMAGES'
    # FILES = ['133_cvc.jpg']

    FILES = ['133_cvc.jpg','419_cvc.jpg','480_cvc.jpg','cju1cfhyg48bb0799cl5pr2jh_kev.jpg']
    MODEL_NAME = 'UNET_no_preprocess_with_crop_v3'
    evaluatemodel = EvaluateModel(FOLDER, FILES, MODEL_NAME,
                                  posprocessing=False)
    evaluatemodel.save_history()
    # evaluatemodel.save_masks()
    # evaluatemodel.get_performance()
    print('fim')