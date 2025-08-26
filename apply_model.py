import cv2
from complement_functions import *
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#######################################################
FOLDER = 'TEST_IMAGES'
FILE = '133_cvc.jpg'
MODEL_NAME = 'UNET_no_preprocess_with_crop_v3'
#######################################################

model_name = '{}.h5'.format(MODEL_NAME)
history_name = 'history/{}.csv'.format(MODEL_NAME)

path_img = './datasets/{}/images/{}'.format(FOLDER,FILE)
path_mask = './datasets/{}/masks/{}'.format(FOLDER,FILE)

# for i in [path_img,path_mask]:
    # img = cv2.imread(i)
    # cv2.imshow(i,img)

# LOAD MODEL
model = tf.keras.models.load_model('./models/{}'.format(model_name),custom_objects={'dice_loss':dice_loss,'ThresholdLayer': ThresholdLayer})

array = get_files([path_img],type_of_file='image')
prediction = model.predict(array)

binario = np.where(prediction > 0.5, 1, 0)
print(binario.shape)



# plt.imshow(binario[0], interpolation='nearest')
# plt.show()

imagem_binaria = binario[0]

# Converte a imagem binária para o tipo de dados uint8
imagem_binaria = (imagem_binaria * 255).astype(np.uint8)

# Define o kernel (elemento estruturante) para a operação de fechamento
kernel = np.ones((5,5), np.uint8)

# Aplica a operação de fechamento
fechamento = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)

# Exibe a imagem original e a imagem após o fechamento morfológico
plt.subplot(1, 2, 1)
plt.imshow(imagem_binaria, cmap='gray')
plt.title('Imagem binária original')

plt.subplot(1, 2, 2)
plt.imshow(fechamento, cmap='gray')
plt.title('Imagem após o fechamento morfológico')

plt.show()



# kernel = np.ones((5,5),np.uint8)
# binario = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel)

# history_table = pd.read_csv(history_name)
# history_class = GetHistory(history_table)
# history_class.accuracy_vs_val_accuracy()
# history_class.loss_vs_val_loss()

cv2.waitKey(0)
cv2.destroyAllWindows()
