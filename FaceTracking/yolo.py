from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

class FaceTracker_YOLO:
    def __init__(self, video_path, output_path='tracked_video.mp4'):
        self.video_path = video_path
        self.output_path = output_path

    def track_faces(self):
        cap = cv2.VideoCapture(self.video_path)
        frames_tracked = []
        
        model = YOLO('yolov8m-face.pt')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Tracking frames")

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(frame_pil)
            
            for info in results:
                if info.boxes is not None:
                    for box in info.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 6)
                        
            frames_tracked.append(frame)
            progress_bar.update(1)

        cap.release()
        progress_bar.close()

        dim = frames_tracked[0].shape[:2][::-1]
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        video_tracked = cv2.VideoWriter(self.output_path, fourcc, 25.0, dim)

        for frame in frames_tracked:
            video_tracked.write(frame)

        video_tracked.release()

