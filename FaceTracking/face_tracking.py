from mtcnn import FaceTracker_MTCNN
from yolo import FaceTracker_YOLO

tracker_MTCNN = FaceTracker_MTCNN(video_path='videos/people.mp4', output_path='videos/people_tracked_MTCNN.mp4')
tracker_MTCNN.track_faces()

tracker_YOLO = FaceTracker_YOLO(video_path='videos/people.mp4', output_path='videos/people_tracked_YOLO.mp4')
tracker_YOLO.track_faces()