import cv2
import os


def write_video_block(images_path):
    """
    Writes a block of images from the specified path to a video file.
    Args:
        images_path (str): The path to the directory containing the images.
    """
    image_files = sorted(
        [os.path.join(images_path, image_filename) for image_filename in os.listdir(images_path)
         if image_filename.endswith(".jpg") or image_filename.endswith(".png")],
        key=os.path.getmtime)  # sort by modification time
    
    if not image_files:
        print(f"Error: No images found in directory: {images_path}")
        return
    
    # get frame dimensions from the first image
    first_image = cv2.imread(image_files[0])
    frame_height, frame_width, _ = first_image.shape

    # create the VideoWriter object 
    video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4.0, (frame_width, frame_height))

    # read and write images to the video
    for image_filename in image_files:
        image = cv2.imread(image_filename)
        if image is None:
            print(f"Error reading image: {image_filename}")
            continue
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()


if __name__ == '__main__':
  print("This code will only run when the script is executed directly.")
  
  images_folder = '/home/aaeon/new_env_FD/results/frames_procss'
  write_video_block(images_folder)
