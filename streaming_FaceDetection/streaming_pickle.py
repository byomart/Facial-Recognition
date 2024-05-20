import os
import cv2
import yaml
import time
import copy
import torch
import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

from auxiliar import (detect_faces_yolov8, crop_faces, calculate_embeddings,
                      get_best_distance, draw_unknown, draw_suspicious, 
                      split_prediction, CameraBufferCleanerThread)

logging.basicConfig(filename = 'logs/streaming_pickle.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level='INFO')


# RUTAS
API_path = os.getcwd() 
suspicious_pickle = os.path.join(API_path, "suspicious.pickle")
output_folder = 'results/results_with_boxes'
os.makedirs(output_folder, exist_ok=True)
images_folder = '/home/aaeon/new_env_FD/results/frames_procss'



# CONFIG.YML
with open(os.path.join(API_path, 'config.yml'), 'r') as config_file:
    # Diccionario con todas las variables del fichero de configuracion
    YML = yaml.safe_load(config_file)


# MODELOS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Deteccion de caras
model_faces_path = os.path.join(API_path, "models", YML["models"]["face_detection"])
model_faces = YOLO(model_faces_path)
# Extraccion de embeddings
model_embeddings = InceptionResnetV1(pretrained='vggface2', device = 'cuda').eval()


# THRESHOLD DISTANCIA ENTRE EMBEDDINGS
threshold = YML["threshold"]

# RECTANGULO SOBRE STREAMING
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor_green = YML["font"]["green"]
fontColor_red = YML["font"]["red"]
fontScale = YML["font"]["size"]

# 'Desconocido' POR PANTALLA CUANDO NO ESTA EN BBDD
text_unknown = YML["unknown_face"]




############################################################################################
################################# STREAMING PROCESSING ######################################
##############################################################################################

# se abre webcam local y se establece tamaño de la ventana 
source ="rtsp://root:Cervustrc*@192.168.3.3:554/axis-media/media.amp" 
cap = cv2.VideoCapture(source)


frame_counter = 0
buffer_size = 10  # bloque con n frames que va al video.writer
buffer_frames = []
while True:

    ######################################### START TIMER ########################################
    iteration_start_time = time.time()
    ##############################################################################################
    
    ret, frame = cap.read() # ret es booleano, img es una matriz numpy
    frame_copy = copy.copy(frame)
    height, width, _ = frame.shape
    print(frame.shape)
    print(f"frame_{frame_counter}")
    
    if not ret:
        break

    # embeddings de todas las caras detectadas en las frames del streaming
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detect_faces_yolov8(frame_pil, model_faces)
    cropped_faces, coord_list = crop_faces(frame_pil, results)
    new_embeddings = calculate_embeddings(cropped_faces, model_embeddings, device) # torch.Size([1, 512])

    # calculo de distancia entre embeddings
    for i, coord in zip(new_embeddings, coord_list):
        for emb in i:
            output_preds, probability = get_best_distance(suspicious_pickle, emb, threshold)
            print('Face Name:', output_preds)

            # dibujo boxes rojos o verdes en funcion de si esta en la base de datos o no
            if output_preds == 'Desconocido':
                # rectangulo verde
                draw_unknown(frame_copy, font, fontColor_green, fontScale, coord[0], coord[1], coord[2], coord[3], text_unknown)
            else:
                # rectangulo rojo con nombre
                name, _, _ = split_prediction(output_preds)
                draw_suspicious(frame_copy, font, fontColor_red, fontScale, coord[0], coord[1], coord[2], coord[3], name)

    filename = os.path.join('results/frames_procss', f"frame_{frame_counter}.jpg")
    cv2.imwrite(filename, frame_copy)

    if cv2.waitKey(1) == ord('q'): # exit en tecla 'q'
        break
    
    frame_counter += 1
    
    ######################################### STOP TIMER #########################################
    iteration_end_time = time.time()
    iteration_time = iteration_end_time - iteration_start_time
    print(f"Iteration time: {iteration_time:.2f} seconds")
    ##############################################################################################

cap.release()
cv2.destroyAllWindows()
