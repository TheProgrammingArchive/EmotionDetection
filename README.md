# EmotionDetection
### High school project on machine learning using TensorFlow, Keras and OpenCV
<b> A model that recognizes emotions and classifies them into 7 categories.

<b> Dataset used (emotion detection) </b>: Fer13
<br>
<b> Final model accuracy (emotion detection) </b>: 74.5% on training set, 64% on test set
<br>
<br>
<b> Dataset used (age detection) </b>: UTKFace
<br>
<b> Final model mae (age detection) </b>: 5.6


### Download the models here: </i><br>
  <b> Age Detection model <b>:  https://drive.google.com/file/d/12WBK04uGUh4yYU8WzYbY7ZNapQ6gkGcv/view?usp=share_link <br>
  <b> Emotion Detection model <b>: https://drive.google.com/file/d/1ncwAS_jwq__WJd8pBDNyuLx35pGJtilM/view?usp=share_link


### Setup: <br>
  * Clone the repository
  * Download both models and place them in the 'EmotionDetection/' directory
  * Make sure dependencies are installed
  
### Usage: <br>
  <b>To run using streamlit: </b><br>
  * streamlit run app.py

  <b>To run using terminal: </b><br>
  * python detector.py -w (for windowed detection mode)
  * python detector.py -c (for camera detection mode)
  
### Dependencies: <br>
  * TensorFlow, Opencv, Pillow
  * streamlit (optional)
