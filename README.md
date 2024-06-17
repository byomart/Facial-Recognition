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

Now, we will explore and compare two popular approaches for real-time object and face tracking: YOLOv8 (You Only Look Once, version 8) and MTCNN (Multi-task Cascaded Convolutional Networks). Both methods have their own advantages and applications in the field of computer vision. Through this comparison, we will analyze the accuracy, efficiency and ease of implementation of each technique.


- **MTCNN (Multi-Task Cascaded Convolutional Neural Networks)**


<p align="center">
<img width="550" alt="2" src="https://github.com/fbayomartinez/Facial-Recognition/blob/2bcea7ea77adb9aa00cdf17aa33c1ad94ead4d0d/FaceTracking/arch_MTCNN.png">
</p>


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/545fd310-41af-4b6b-b978-373dfddfe940" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>



- **Yolov8-face**


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/911c4b6b-8928-460a-9c35-bfacbdd5538a" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>


Our evaluation shows that YOLOv8 offers a more reliable approach and achieves better results compared to MTCNN. YOLOv8 stands out for its fast and accurate detection capability, maintaining an ideal balance between speed and accuracy. This makes it especially useful in real-time tracking applications where reliability and accuracy are critical. On the other hand, MTCNN, while effective in face detection, may not match the robustness and versatility of YOLOv8 in a broader tracking environment.



## Face Recognition on video:

Next we will perform a detailed comparison between two of the most widely used datasets in the field of face recognition: CASIA-WebFace and VGG-Face. The objective is to determine which of these datasets provides better results when used with the Inception Resnet model. 

- **vgg2face**


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/d6b24821-c4b3-46d8-a865-1c388aaa392a" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>



- **casia-webface**


<p align="center">
  <img src="https://github.com/fbayomartinez/Facial-Recognition/assets/163590683/aa7b074c-38df-4191-ac8b-756cf8ff77e5" alt="people_tracked_MTCNN-ezgif com-video-to-gif-converter" width="550">
</p>

When using Inception Resnet (IR) models in pytorch, pretrained on VGGFace2 and CASIA-Webface, first provides higher accuracy and better overall performance compared to CASIA-WebFace. VGGFace2, known for its wide diversity and high quality of facial images, enables to achieve higher levels of reliability and accuracy in face recognition tasks. On the other hand, although CASIA-WebFace is a robust and extensive dataset, the results suggest that its performance with IR does not reach the same level of effectiveness as VGGFace2.





<!--
## Face Recognition on real-time streaming:

- API
- Streaming_Pickle.py: **Yolov8** + **Inception Resnet V1** (all running over pythorch)
- Frames 2 video (processed)
-->

