from fastapi import FastAPI, UploadFile, File
import logging
import time
import os

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


from auxiliar import (save_local_file, detect_and_crop_faces, calculate_embeddings_susp,
                      register_new_embeddings, read_pickle, delete_element, 
                      delete_all_embeddings, check_pickle)

logging.basicConfig(filename = 'logs/suspicious.log',
                    filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level='INFO')


# Rutas
API_path = os.getcwd() # '\facial_int\env_face_int\IRV1_pytorch\streaming_API'
suspicious_pickle = os.path.join(API_path, "suspicious.pickle")
save_cropped_path = os.path.join(API_path, "cropped_faces")
pickle_path = "suspicious.pickle"



# Modelos
model_faces = YOLO('models/yolov8m-face.pt')
model_embeddings = InceptionResnetV1(pretrained='vggface2').eval()



# # Reset pickle file if needed
# delete_pickle(pickle_path)


app = FastAPI()

# AÑADIR SOSPECHOSOS A LA BASE DE DATOS
@app.post("/save_suspicious")
def insert_suspicious(nombre: str = '', dni: str = '', motivo: str = '',  file: UploadFile = File(...)):
    
    if not file:
        return {"message": "No upload file sent"}
   
    else:
        tstart = time.time()

        # Se guarda la imagen introducida por la API
        folder_temp = 'images_temp'
        save_temp_path = save_local_file(folder_temp, file, API_path)

        # Se extrean embeddings de las caras presentes en la imagen (lo ideal es que solo aparezca una en la introducción de un sospechoso) y se meten en el pickle
        suspicious_face = detect_and_crop_faces(model_faces, save_temp_path, save_cropped_path)
        suspicious_face_embedding = calculate_embeddings_susp(model_embeddings, suspicious_face)
        register_new_embeddings(suspicious_pickle, nombre, dni, motivo, suspicious_face_embedding)
        
        # Mostrar info en .pickle (Nombre_DNI_Motivo: embedding)
        pickle_info = read_pickle(pickle_path)
        logging.info(f'Pickle {pickle_info}')
        logging.info(f"Se han empleado {(time.time() - tstart):.3f} segundos en extraer y guardar embeddings de las caras en la foto: {file.filename} ")
    
        return {"message": f"Los embeddings de {nombre}, con DNI {dni} y motivo {motivo} se han guardado correctamente"}


# BORRAR UN SOSPECHOSO DE LA BASE DE DATOS
@app.post("/deleteONE")
def delete_suspicious(nombre: str = '', dni: str = '', motivo: str = ''):
    return delete_element(nombre, dni, motivo, suspicious_pickle)


# CHEQUEAR SOSPECHOSO DE LA BASE DE DATOS
@app.post("/checkBBDD")
def check_BBDD():
    faces_dict = check_pickle(suspicious_pickle)
    return list(faces_dict)


# BORRAR LA BASE DE DATOS
@app.post("/deleteALL")
def delete_all_suspicious():
    return delete_all_embeddings(suspicious_pickle)

