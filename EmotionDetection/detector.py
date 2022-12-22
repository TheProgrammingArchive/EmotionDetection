from keras.models import load_model
import numpy as np
import cv2
from keras.utils import img_to_array
from keras.preprocessing import image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
prediction_model = load_model('model')

capture = cv2.VideoCapture(0)

emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
colors = {0:(0, 128, 255), 1:(246, 61, 252), 2:(246, 61, 252), 3:(0,255,90), 4:(204, 204, 204), 5:(237, 28, 63), 6:(252,238,33)}

while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi]) != 0:
            roi_ = roi.astype('float')/255.0
            roi_ = img_to_array(roi_)
            print(roi_.ndim, roi_.shape)
            roi_ = np.expand_dims(roi_, axis=0)
            print(roi_.shape)

            prediction = prediction_model.predict(roi_)[0]
            emotion = emotion_labels[prediction.argmax()]
            label_pos = (x, y - 10)
            cv2.putText(frame, emotion, label_pos, cv2.FONT_HERSHEY_PLAIN, 2, colors[prediction.argmax()], 2)

        else:
            pass

    cv2.imshow('Detector', frame)

    if cv2.waitKey(1) == ord('Q'):
        break

cv2.destroyAllWindows()