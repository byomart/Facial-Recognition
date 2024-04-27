import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch


def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            images.append(img)
    return images


def draw_boxes(image, boxes, box_output_folder, cropped_output_folder, filename, target_size=(160, 160)):
    '''
    Function for drawing bounding boxes around faces in a given image. It then crops these areas 
    and saves the resulting images both with the bounding boxes and by themselves in separate folders. 
    
    After saving cropped images, we normalize [-1,1].
    '''
    
    os.makedirs(box_output_folder, exist_ok=True)
    os.makedirs(cropped_output_folder, exist_ok=True)

    output_path = os.path.join(box_output_folder, filename)
   
    fig, ax = plt.subplots()

    cropped_tensors_list = [] 
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        cropped_img = image.crop((x1, y1, x2, y2))
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        cropped_tensor = transform(cropped_img)
        cropped_tensors_list.append(cropped_tensor)

    ax.imshow(image)
    plt.savefig(output_path)
    plt.close()

    for i, cropped_tensor in enumerate(cropped_tensors_list):
        cropped_output_path = os.path.join(cropped_output_folder, f'{filename}_cropped_{i}.png')
        plt.imshow(cropped_tensor.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(cropped_output_path)
        plt.close()

    # normalization between [-1,1] 
    normalized_cropped_img_list = []
    for cropped_tensor in cropped_tensors_list:
        normalized_cropped_img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(cropped_tensor)
        normalized_cropped_img_list.append(normalized_cropped_img)

    return normalized_cropped_img_list


def compute_embedding(model, tensor):
    face_embeddings = model(tensor).detach()

    return face_embeddings


def most_similar_face(test_face_embedding, faces_embedding):
    '''
    Find the most similar face in a set of face embeddings.
    '''
    similarities = F.cosine_similarity(test_face_embedding, faces_embedding, dim=1)
    distances = 1 - similarities
    min_distance = torch.min(distances)
    min_distance_index = torch.argmin(distances)

    return distances, min_distance_index, min_distance