from torchvision import transforms
import torch.nn.functional as F
import cv2
import os

class FaceDetector:
    def __init__(self, model_faces):
        self.model_faces = model_faces

    def detect_faces(self, frame_pil):
        return self.model_faces(frame_pil)

    def crop_faces(self, frame_pil, results):
        cropped_faces = []
        for i, info in enumerate(results):
            if info.boxes is not None:
                for box in info.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cropped_img = frame_pil.crop((x1, y1, x2, y2))
                    cropped_faces.append(cropped_img)
        return cropped_faces
    


class EmbeddingsCalculator:
    def __init__(self, model_embeddings):
        self.model_embeddings = model_embeddings

    def calculate_embeddings(self, faces):
        embeddings = []
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for face in faces:
            face_tensor = transform(face)
            face_embedding = self.model_embeddings(face_tensor.unsqueeze(0)).detach()
            embeddings.append(face_embedding)
        return embeddings



class DistanceCalculator:
    def calculate_distances(self, new_embeddings, database_embeddings):
        all_distances = []
        for new_embedding in new_embeddings:
            distances_per_box = []
            for db_embedding in database_embeddings:
                similarities = F.cosine_similarity(new_embedding, db_embedding, dim=1)
                distance = 1 - similarities
                distances_per_box.append(distance)
            all_distances.append(distances_per_box)
        return all_distances

    def draw_faces(self, frame, results, distances, output_folder, frame_count):
        for info in results:
            if info.boxes is not None:
                for i, box in enumerate(info.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    distances_for_box = distances[i]
                    min_distance = min(distances_for_box)
                    
                    if min_distance < 0.45:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)  # rojo si distancia < umbral
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)  # verde si distancia > umbral
                        
                cv2.imwrite(os.path.join(output_folder, f'imagen_guardada_{frame_count}.png'), frame)


