import os
import cv2
import pickle
import logging
import threading
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.spatial import distance


############################################################################################
################################# API FUNCTIONS ############################################
############################################################################################

def save_local_file(folder_temp, file, api_path):
    '''
    Funcion que guarda una imagen en una carpeta local (que se crea si no está creada) y devuelve la 
    ruta global de dicha imagen
    - Entrada:
    folder_temp: Nombre de la carpeta que se crea (generalmente temporal) para almacenar esta imagen
    file: Objeto archivo con varios campos (el nombre y el archivo .jpg ente otros)
    api_path: Ruta inicial de la api
    - Salida:
    save_temp_path: Ruta global del archivo que se ha guardado 
    '''
    creation_folder(folder_temp)

    save_temp_path = os.path.join(api_path, folder_temp, f'{file.filename}')
    with open(save_temp_path, "wb") as f:
        f.write(file.file.read())
    return save_temp_path


def register_new_embeddings(data_pickle, nombre, dni, motivo, embeddings_list):
    '''
    Funcion que introduce los embeddings de una cara en un pickle con el nombre, dni y motivo indicados
    - Entradas:
    data_pickle: Ruta del pickle donde se introduce el diccionario completo
    nombre: Nombre del sospechoso
    dni: DNI del sospechoso
    motivo: Motivo por el que se introduce a un sospechoso
    embeddings_list: Lista de N embeddings de la misma cara (aumentada N-1 veces)
    - Salida:
    No hay, solo se introduce la informacion en el pickle
    '''
    # Se crea el diccionario que se introduce en el pickle 
    # if os.stat(os.path.join(os.getcwd(), data_pickle)).st_size == 0:
    if os.stat(data_pickle).st_size == 0:
        embds_dict = {}
    else:
        embds_dict = pickle.loads(open(data_pickle, "rb").read())
    # Introducir los nuevos embeddings
    embds_dict[f'{nombre}_{dni}_{motivo}'] = embeddings_list
    # Save dictionary
    with open(os.path.join(os.getcwd(), data_pickle), 'wb') as tf:
        pickle.dump(embds_dict, tf)


def detect_and_crop_faces(model_faces, save_temp_path, save_cropped_path):
    """
    Detecta y recorta las caras en un frame 
    - Entradas:
    model_faces: Modelo de detección de caras
    frame_pil: Imagen del frame convertida a formato PIL
    save_temp_path: Carpeta 'temporal' con la imagen del nuevo sospechoso a registrar
    save_cropped_path: Carpeta donde se almacenan todas las caras detectadas y recortadas
    - Salida:
    cropped_faces: Lista de imágenes recortadas de las caras detectadas
    """
    # se crea el directorio en caso de no existir
    if not os.path.isdir(save_cropped_path):
            os.makedirs(save_cropped_path)

    # se convierte la imagen a formato PIL
    image_numpy = cv2.imread(save_temp_path)
    frame_pil = Image.fromarray(cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB))
    
    # deteccion de caras
    results = model_faces(frame_pil)

    # recorte de caras
    areas = []
    cropped_faces = []
    for i, info in enumerate(results):
        if info.boxes is not None:
            for j, box in enumerate(info.boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped_img = frame_pil.crop((x1, y1, x2, y2))

                # Calculo del área de cada box
                area = box_area(x1, x2, y1, y2)
                areas.append(area) 

                # Se guardan los recortes de TODAS las caras detectadas
                file_name = f"cropped_face_{j}.jpg"
                file_path = os.path.join(save_cropped_path, "all_cropped_faces", file_name)
                cropped_img.save(file_path)

                # Lista con el recorte de todas las caras detectadas
                cropped_faces.append(cropped_img)

    # Selección de la cara del sospechoso (box con mayor área)
    final_suspicious_index = areas.index(max(areas))
    final_suspicious = cropped_faces[final_suspicious_index]
    file_path = os.path.join(save_cropped_path, "final_suspicious", "suspicious.jpg")
    final_suspicious.save(file_path)

    return final_suspicious


def calculate_embeddings_susp(model_embeddings, face):
    '''
    Calculo de embedings de una cara detectada
    - Entradas:
    model_embeddings: Modelo de calculo de embeddings
    face: Imagen de la cara sobre la que calcular embedding
    - Salida:
    face_embedding: Embedding de la cara introducida.
    '''
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_tensor = transform(face)
    face_embedding = model_embeddings(face_tensor.unsqueeze(0)).detach()

    return face_embedding


def box_area(x1, x2, y1, y2):
    """  límites (x1, x2, y1, y2).
    """
    base = abs(x2 - x1)
    altura = abs(y2 - y1)
    area = base * altura
    return area


def read_pickle(pickle_file):
    with open(pickle_file, "rb") as archivo:
        pickle_data = pickle.load(archivo)
    return pickle_data

def delete_pickle(pickle_file):
    with open(pickle_file, 'wb') as archivo_pickle:
        pickle.dump({}, archivo_pickle)


def creation_folder(folder):
    '''
    Funcion que crea una carpeta en el directorio actual, en caso de que esta no exista
    - Entrada: 
    folder: Nombre de la carpeta
    - Salida:
    No hay, se crea la carpeta
    '''
    if not os.path.isdir(folder):
        os.mkdir(folder)
        

def delete_element(nombre, dni, motivo, data_pickle):
    '''
    Funcion que elimina un unico sospechoso del pickle de sospechosos
    - Entrada:
    nombre: Nombre del sospechoso a eliminar del pickle
    dni: DNI del sospechoso a eliminar del pickle
    motivo: Motivo del sospechoso a eliminar del pickle
    data_pickle: Ruta del pickle donde se encuentra el diccionario completo
    - Salida:
    Unicamente hay un mensaje exitoso o de error en funcion de si se han introducido valores coherentes
    '''
    if os.stat(os.path.join(os.getcwd(), data_pickle)).st_size == 0:
        embds_dict = {}
        return {"message": f"No hay ninguna cara sospechosa en la base de datos"}
    else:
        embds_dict = pickle.loads(open(os.path.join(os.getcwd(), data_pickle), "rb").read())
        identification = f'{nombre}_{dni}_{motivo}'
        if identification in embds_dict:
            del embds_dict[identification]
            with open(os.path.join(os.getcwd(), data_pickle), 'wb') as tf:
                pickle.dump(embds_dict, tf)
            # Se comprueba leyendo el pickle
            known_faces, known_names, known_embeddings = obtain_pickle_dict(data_pickle)
            print (known_faces)
            print ("Total de caras conocidas:", len(known_names))
            print ("Nombres de las caras:", known_names)
            print ("Dimensiones del array de embeddings:", np.array(known_embeddings[0]).shape)
            return {"message": f"La cara sospechosa de {nombre}, con dni {dni} y motivo {motivo} ha sido eliminada"}
        else:
            return {"message": f"La cara sospechosa de {nombre}, con dni {dni} y motivo {motivo} no se encuentra en la base de datos. Algun campo introducido no es correcto"}



def delete_all_embeddings(data_pickle):
    '''
    Funcion que elimina todos los sospechosos del pickle de sospechosos
    - Entrada:
    data_pickle: Ruta del pickle donde se encuentra el diccionario completo
    - Salida:
    Unicamente hay un mensaje exitoso o de error en funcion de si habia o no elementos en la BBDD
    '''
    if os.stat(data_pickle).st_size == 0:
        embds_dict = {}
        return {"message": f"No hay ninguna cara sospechosa en la base de datos"}
    else:
        # Se inicializa un diccionario vacío
        embds_dict = {}
        # Se guarda el diccionario creado
        with open(data_pickle, 'wb') as tf:
            pickle.dump(embds_dict, tf)
        return {"message": f"Se ha eliminado toda la BBDD, de hecho, se muestra {embds_dict}"}
    

def check_pickle(data_pickle):
    '''
    Funcion que comprueba los datos almacenados en el pickle de sospechosos
    - Entrada:
    data_pickle: Ruta del pickle donde se encuentra el diccionario completo
    - Salida:
    known_faces: Diccionario completo
    '''
    if os.stat(data_pickle).st_size == 0:
        known_faces = {}
    else:
        known_faces = pickle.loads(open(data_pickle, "rb").read())
    return known_faces





############################################################################################
#################################### STREAMING FUNCTIONS ###################################
############################################################################################

def detect_faces_yolov8(frame_pil, detection_model):
    return detection_model(frame_pil)

def crop_faces(frame_pil, results):
    cropped_faces = []
    coords_list = []
    for i, info in enumerate(results):
        if info.boxes is not None:
            for box in info.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped_img = frame_pil.crop((x1, y1, x2, y2))
                cropped_faces.append(cropped_img)
                coords_list.append((x1, y1, x2, y2))
    return cropped_faces, coords_list 

# def calculate_embeddings(faces, model_embeddings):
    
#     embeddings = []
#     transform = transforms.Compose([
#         transforms.Resize((160, 160)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     for face in faces:
#         face_tensor = transform(face)
#         face_embedding = model_embeddings(face_tensor.unsqueeze(0)).detach()
#         embeddings.append(face_embedding)
#     return embeddings


def calculate_embeddings(faces, model_embeddings, device):
    
    embeddings = []
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Move all faces to the chosen device before the loop
    faces_tensors = [transform(face).to(device) for face in faces]

    for face_tensor in faces_tensors:
        with torch.no_grad():
            face_embedding = model_embeddings(face_tensor.unsqueeze(0)).detach()
        embeddings.append(face_embedding)
    return embeddings



def obtain_pickle_dict(data_pickle):
    known_faces = pickle.loads(open(data_pickle, "rb").read())
    known_names = list(known_faces.keys())
    known_embeddings = list(known_faces.values())
    return known_faces, known_names, known_embeddings


def get_best_distance(data_pickle, embeddings, error_threshold):
        '''
        Function for obtaining the lowest distance of an array of embeddings to the users saved in the database
        '''
        output_preds = None
        output_min_distance = None
        known_faces, known_names, _ = obtain_pickle_dict(data_pickle)
        try:
            min_distance = 5
            pred = None
            for known_name in known_names:
                for embds in known_faces[str(known_name)]:
                    
                    # pasamos el tensor de embeddings de caras detectadas a cpu
                    embeddings_cpu = embeddings.clone().cpu()
       
                    embds_cos_distance = distance.cosine(embds,embeddings_cpu)

                    if embds_cos_distance < min_distance:
                        min_distance = embds_cos_distance
                        output_preds = str(known_name)        
                    output_min_distance = min_distance
                    # If there is noy enough similarity, do NOT return any known face 
                    if output_min_distance < error_threshold: 
                        probability = 1- output_min_distance
                    else:
                        probability = 0
                        output_preds = 'Desconocido'    
            return output_preds, probability
        except Exception as ex:
            logging.error(ex)
            return "",-1


def draw_unknown(frame_copy, font, fontColor, fontScale, x1, y1, x2, y2, name):
    position = (x1, y1)
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), fontColor, 4)
    cv2.putText(frame_copy, name, position, font, fontScale, fontColor, thickness = 3)


def draw_suspicious(frame_copy, font, fontColor, fontScale, x1, y1, x2, y2, name):
    position = (x1, y1)
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), fontColor, 4)
    cv2.putText(frame_copy, name, position, font, fontScale, fontColor, thickness = 3)

def split_prediction(output_preds):
    name = output_preds.split("_")[0]
    dni = output_preds.split("_")[1]
    motivo = output_preds.split("_")[2]
    return name, dni, motivo


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()