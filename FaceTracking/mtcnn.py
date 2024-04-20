from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

class FaceTracker_MTCNN:
    def __init__(self, video_path, output_path='tracked_video.mp4'):
        self.video_path = video_path
        self.output_path = output_path

    def track_faces(self):
        cap = cv2.VideoCapture(self.video_path)

        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame_pil)

        cap.release()

        print(len(frames))

        mtcnn = MTCNN(keep_all=True)

        frames_tracked = []
        for i, frame in enumerate(tqdm(frames, desc="Tracking frames")):
            boxes, _ = mtcnn.detect(frame)

            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            if boxes is not None:
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))

        dim = frames_tracked[0].size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        video_tracked = cv2.VideoWriter(self.output_path, fourcc, 25.0, dim)

        for frame in frames_tracked:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()


