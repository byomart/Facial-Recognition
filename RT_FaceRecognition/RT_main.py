import os
import cv2
import logging
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

from RT_read_ddbb import read_DDBB
from RT_read_video import FaceDetector, EmbeddingsCalculator, DistanceCalculator
from RT_img2vid import imgseq2video

logging.basicConfig(filename = 'logs/bardem2_POO_2_casiawebface.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level='INFO')

# models
model_faces = YOLO('yolov8m-face.pt')
#model_embeddings = InceptionResnetV1(pretrained='vggface2').eval()
model_embeddings = InceptionResnetV1(pretrained='casia-webface').eval()


# paths
path_ddbb = 'Faces_BBDD_cropped'
path_video = 'videos/bardem2.mp4'
output_folder = 'results_with_boxes_POO'
os.makedirs(output_folder, exist_ok=True)
output_video_path = 'output_video_bardem2_casiaweface.mp4'

# load database
ddbb = read_DDBB(model_embeddings)
raw_images = ddbb.read_images(path_ddbb)
tensor_ddbb = ddbb.transform_images(raw_images)
embeddings_ddbb = ddbb.calculate_embeddings(tensor_ddbb)


# load new video
face_detector = FaceDetector(model_faces)
embeddings_calculator = EmbeddingsCalculator(model_embeddings)
distance_calculator = DistanceCalculator()

frame_count = 0
all_distances = []
cap = cv2.VideoCapture(path_video)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = face_detector.detect_faces(frame_pil)
    cropped_faces = face_detector.crop_faces(frame_pil, results)
    new_embeddings = embeddings_calculator.calculate_embeddings(cropped_faces)
    distances = distance_calculator.calculate_distances(new_embeddings, embeddings_ddbb)
    sorted_distances = [sorted(distance, key=lambda x: x.item()) for distance in distances]
    distance_calculator.draw_faces(frame, results, distances, output_folder, frame_count)

    # all_distances.extend(distances)
    logging.info(f'Frame: {frame_count}')
    for i, face_distances in enumerate(sorted_distances):
        logging.info(f'Face {i} - distances: {face_distances}')

    frame_count += 1
cap.release()

# save image sequence as new tracked video
video_creator = imgseq2video(output_folder, output_video_path)
video_creator.create_video()

