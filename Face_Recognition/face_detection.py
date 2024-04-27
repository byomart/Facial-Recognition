from ultralytics import YOLO
import torch
import logging
from utils import read_images, draw_boxes
import os


def detect_faces(folder_path, test_img_output_path = None):
    '''
    Process a folder of images, detecting and cropping faces within them

    Args: folder_path (str): path to the folder containing input images

    Returns: tensor containing cropped face images from all input images
            The tensor has shape [N, C, H, W], where:
            - N is the total number of cropped face images
            - C is the number of channels (3 for RGB)
            - H is the height of each image
            - W is the width of each image
    '''

    logging.basicConfig(filename='logs/log.log', level=logging.INFO, filemode='w')
    detection_model = YOLO('yolov8m-face.pt')

    # input images
    raw_images = read_images(folder_path)

    cropped_imgs_list = []
    if len(raw_images) > 1:
        for i, raw_img in enumerate(raw_images):
            results = detection_model.predict(raw_img)
            cropped_imgs = draw_boxes(raw_img, results[0].boxes, 'images/database/boxed_images', 'images/database/cropped_images', f'image_{i}', target_size=(160, 160))
            cropped_imgs_list.append(torch.stack(cropped_imgs)) 

            logging.info(f' Raw image {i+1} size: {raw_img.size}')
            logging.info(f' Faces detected within image {i+1}: {len(cropped_imgs)}')
            logging.info(f' Dimension of each cropped image saved from image {i+1}: {cropped_imgs[0].shape}')

    elif len(raw_images) == 1 and test_img_output_path:
        results = detection_model.predict(raw_images[0])
        cropped_imgs = draw_boxes(raw_images[0], results[0].boxes, 'images/test/boxed_images', test_img_output_path, f'image_test', target_size=(160, 160))
        cropped_imgs_list.append(torch.stack(cropped_imgs))

    cropped_imgs_tensor = torch.cat(cropped_imgs_list, dim=0)
    logging.info(f' [Faces, RGB, X, Y]: {cropped_imgs_tensor.shape}')

    return cropped_imgs_tensor










