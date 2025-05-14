import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



model = load_model('messi_binary_classifier.h5')
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Nie można załadować kaskady z {cascade_path}")

img = cv2.imread('resources/picture.jpg')
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50,50)
)

for (x, y, w, h) in faces:
    crop = orig[y:y+h, x:x+w]

    roi = cv2.resize(crop, (200,200)).astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)   

    p_other = model.predict(roi)[0][0]
    p_messi = 1 - p_other
    print(f"p(messi) = {p_messi:.3f}, p(other) = {p_other:.3f}")

    if p_messi >= p_other:
        label, conf, color = 'messi', p_messi, (0,255,0)
    else:
        label, conf, color = 'other', p_other, (0,0,255)

    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv2.putText(
        img,
        f"{label} ",
        # ({conf:.2f})
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

cv2.imwrite('output.jpg', img)
print("Zapisano oznaczone zdjęcie jako output.jpg")
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
