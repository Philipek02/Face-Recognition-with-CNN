import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='Oznaczanie twarzy na wideo (messi vs other)')
parser.add_argument('--video', type=str, default='resources/video.mp4',
                    help='Ścieżka do pliku wideo lub indeks kamery (0 domyślnie)')
args = parser.parse_args()

video_src = int(args.video) if args.video.isdigit() else args.video
cap = cv2.VideoCapture(video_src)
if not cap.isOpened():
    print(f"Nie można otworzyć źródła wideo: {args.video}")
    exit(1)

model = load_model('messi_binary_classifier.h5')
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Nie można wczytać kaskady z {cascade_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50)
    )
    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        roi = cv2.resize(crop, (200,200)).astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)

        p_other = model.predict(roi, verbose=0)[0][0]
        p_messi = 1.0 - p_other

        if p_messi > 0.6:
            label = 'messi'
            conf  = p_messi
            color = (0,255,0)
        else:
            label = 'other'
            conf  = p_other
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#  ({conf:.2f})

    cv2.imshow('Video Face Annotation', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
