# Artificial-Vision

## Face Recognition:


- **Database creation**

First step will be to create a database with the faces against which we want to check the new face. In this way, we will be able to recognize whether a person has already been registered or not. The clearest example for the application of this algorithm is reflected in the detection or identification of suspects, people with a criminal record... 

For this example, we have used images of complete squads of soccer teams, on which we have carried out the detection of faces for future facial recognition.

<p align="center">
<img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/174a269c-b736-44d7-b02e-3fa3f05f62a7" width="330"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/d9936c9a-bdaa-41b7-9ff5-8639cdd84101" width="330"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/fb82c9a7-42b4-490d-92a3-b9f5c1659e5b" width="330"/>
</p>
<br>

- **Testing new image**

Once the database has been created, we will check the correct functioning of the algorithm and, to do so, we will introduce a new image on which we will detect the existing faces.

<p align="center">
<img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/09702d53-337a-4471-bcc5-41bba236b675" width="350"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/5a19f74e-b3c9-4b6b-a6d6-50e94fccb5e8" width="350"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/d13a1d55-a763-4252-8b88-254895f5964a" width="260"/>
</p>
<br>

- **Result:**

Finally, a comparison of the embeddings corresponding to all the faces stored in the database against the embedding of the new test image is carried out, verifying that the most similar face corresponds to the same person (the Xavineta).

<p align="center">
<img width="750" alt="Captura de pantalla 2024-06-17 100212" src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/325dac6b-31fb-4086-9065-4d429da22d91">
</p>

<p align="center">
<img width="416" alt="2" src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/159b0c7b-aadc-40af-8a6c-ffa402e71d8a">
</p>

<br>

## Face Tracking: MTCNN vs Yolov8-face


- **MTCNN (Multi-Task Cascaded Convolutional Neural Networks)**

<p align="center">
<img width="416" alt="2" src="https://github.com/fbayomartinez/Facial-Recognition/blob/2bcea7ea77adb9aa00cdf17aa33c1ad94ead4d0d/FaceTracking/arch_MTCNN.png">
</p>


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/545fd310-41af-4b6b-b978-373dfddfe940" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>



- **Yolov8-face**


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/911c4b6b-8928-460a-9c35-bfacbdd5538a" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>



## Face Recognition on video:

- **vgg2face**

https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/69ee230a-16f4-4d7a-bdd3-74f39000d952


- **casia-webface**
  
https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/7e1337bf-6b7b-43d4-8783-6d27d69e430f






## Face Recognition on real-time streaming:

- API
- Streaming_Pickle.py: **Yolov8** + **Inception Resnet V1** (all running over pythorch)
- Frames 2 video (processed)

