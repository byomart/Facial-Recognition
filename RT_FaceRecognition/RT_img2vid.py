import cv2
import os

class imgseq2video:
    def __init__(self, images_folder, output_video_path, fps=30):
        self.images_folder = images_folder
        self.output_video_path = output_video_path
        self.fps = fps

    def extract_number(self, filename):
        return int(filename.split('_')[-1].split('.')[0])

    def create_video(self):
        file_names = os.listdir(self.images_folder)
        sorted_file_names = sorted(file_names, key=self.extract_number)

        img = cv2.imread(os.path.join(self.images_folder, sorted_file_names[0]))
        height, width, _ = img.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video codec
        video_writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (width, height))

        for image_file in sorted_file_names:
            image_path = os.path.join(self.images_folder, image_file)
            frame = cv2.imread(image_path)
            video_writer.write(frame)

        video_writer.release()