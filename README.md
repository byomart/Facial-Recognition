# Artificial-Vision

## Face Recognition:


- **Database creation**

El primer paso será crear una base de datos con los rostros contra los que queremos contrastar la nueva cara. De esta forma, seremos capaces de reconocer si una persona ya ha sido registrada o no. El ejemplo más claro para la aplicación de este algoritmo se ve reflejado en la detección o identificación de sospechosos, personas con antecedentes... Para este ejemplo, hemos utilizado imagenes de plantillas completas de equipos de fútbol, sobre las que hemos llevado a cabo la detección de caras para un futuro reconocimiento facial. 

<p align="center">
<img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/174a269c-b736-44d7-b02e-3fa3f05f62a7" width="200"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/d9936c9a-bdaa-41b7-9ff5-8639cdd84101" width="200"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/fb82c9a7-42b4-490d-92a3-b9f5c1659e5b" width="200"/>
</p>

- **Testing** new image

Una vez creada la base de datos, lo que haremos será comprobar el correcto funcionamiento del algoritmo y, para ello, introduciremos una nueva imagen sobre la que detectaremos las caras existentes.

<p align="center">
<img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/09702d53-337a-4471-bcc5-41bba236b675" width="200"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/5a19f74e-b3c9-4b6b-a6d6-50e94fccb5e8" width="200"/> <img src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/d13a1d55-a763-4252-8b88-254895f5964a" width="200"/>
</p>

- **Result:**

Para terminar, se lleva a cabo una comparación de los embeddings correspondientes a todas las caras guardadas en la base de datos frente al embedding de la nueva imagen de test, comprobando como la cara más similar se corresponde con la misma persona (la Xavineta).

<p align="center">
<img width="416" alt="2" src="https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/159b0c7b-aadc-40af-8a6c-ffa402e71d8a">
</p>


## Face Tracking: MTCNN vs Yolov8-face

- **MTCNN (Multi-Task Cascaded Convolutional Neural Networks)**
  
https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/b1b58353-5eb1-42b0-a3c1-4a7f26120e2e


- **Yolov8-face**
  
https://github.com/fbayomartinez/Artificial-Vision/assets/163590683/f0ab621a-d921-4138-950c-0e4d548c2d4c


