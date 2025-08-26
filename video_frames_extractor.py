import cv2
import os
import numpy as np

OUT_FORMAT = 'jpg'
OUT_SIZE_Y = 100
OUT_SIZE_X = 50
STEP = 300

MAIN_VIDEO_PATH = './videos/'
VIDEO = 'ESPIRAL_MAGICA.webm'
FOLDER_SAVE = './datasets/espiral_teste'

def convert_size(image,x,y):
    image = cv2.resize(image,(y,x))
    return image

def iterate_video(MAIN_VIDEO_PATH,VIDEO,FOLDER_SAVE,STEP,OUT_FORMAT):
    cap = cv2.VideoCapture(str(MAIN_VIDEO_PATH+'/'+VIDEO))

    if not cap.isOpened():
        print("Erro ao abrir o vídeo")

    if not os.path.exists(FOLDER_SAVE):
        os.makedirs(FOLDER_SAVE)

    count_frame = 0

    # Read until video is completed
    while(cap.isOpened()):

    # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        if count_frame % STEP == 0:
            cv2.imshow('Frame', frame)
            in_name = '.'.join(VIDEO.split('.')[:-1])
            output_filename = str(FOLDER_SAVE+'/'+in_name+'_frame_{}'.format(count_frame)+'.{}'.format(OUT_FORMAT))
            print(output_filename)

            frame = convert_size(frame,OUT_SIZE_X,OUT_SIZE_Y)

            cv2.imwrite(output_filename, frame)
            print("Frame {} salvo".format(count_frame))

        count_frame += 1

    # When everything done, release
    # the video capture object
    cap.release()

    print("{} frames processados, {} frames salvos na pasta {}".format(count_frame,count_frame//STEP,FOLDER_SAVE))

    # Closes all the frames
    cv2.destroyAllWindows()


##############################################################################################################################

iterate_video(MAIN_VIDEO_PATH,VIDEO,FOLDER_SAVE,STEP,OUT_FORMAT)