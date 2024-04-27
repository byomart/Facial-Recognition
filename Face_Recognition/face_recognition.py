from facenet_pytorch import InceptionResnetV1  
import matplotlib.pyplot as plt
import logging
import torch
from utils import compute_embedding, most_similar_face
from face_detection import detect_faces




embedding_model = InceptionResnetV1(pretrained='casia-webface').eval()

########################################################################
############################ DATABASE FACES ############################
########################################################################
folder_path = 'images/database'
cropped_faces_ddbb = detect_faces(folder_path)
faces_embedding = compute_embedding(embedding_model, cropped_faces_ddbb)
logging.info(f' Embedding dimension (DDBB): {faces_embedding.shape}')
########################################################################


########################################################################
############################ NEW FACE IMAGE ############################
########################################################################
folder_path_test = 'images/test'
cropped_new_face = detect_faces(folder_path_test, test_img_output_path = 'images/test/cropped_images')
test_face_embedding = compute_embedding(embedding_model, cropped_new_face)
logging.info(f' Embedding dimension (TEST IMAGE): {test_face_embedding.shape}')
########################################################################



# distance between embeddings
distances, min_distance_index, min_distance = most_similar_face(test_face_embedding, faces_embedding)
logging.info(f'- Most similar face detected is image number {min_distance_index}, with minimum distance of {min_distance}')
logging.info(f'- Distances between image embeddings are: {distances.sort(dim=-1, descending=False)}')

print(distances.sort(dim=-1, descending=False))

# (prueba) para ver que la imagen con menor distancia es la misma persona que la imagen de test
image = cropped_faces_ddbb[min_distance_index].permute(1,2,0) 
plt.imshow(image)
plt.axis('off') 
output_path = 'images/result/most_similar' 
plt.savefig(output_path)
plt.show()
