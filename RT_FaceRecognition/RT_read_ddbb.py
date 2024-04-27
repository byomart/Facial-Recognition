from torchvision import transforms
from PIL import Image
import os


class read_DDBB:
    def __init__(self, model_embeddings):
        self.model_embeddings = model_embeddings

    def read_images(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                img = Image.open(file_path)
                images.append(img)
        return images


    def transform_images(self, images):
        '''
        Function to change images format to tensor, 
        resize to a common size (160x160)
        and normalize between -1 and 1.
        '''
        transformed_images = []
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for img in images:
            transformed_img = transform(img)
            transformed_images.append(transformed_img)
        return transformed_images


    def calculate_embeddings(self, transformed_images):
        embeddings = []
        for img in transformed_images:
            face_embeddings = self.model_embeddings(img.unsqueeze(0)).detach()
            embeddings.append(face_embeddings)
        return embeddings
