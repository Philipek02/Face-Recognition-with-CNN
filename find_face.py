import os
import cv2

classes = ['messi', 'other']

def process_folder(base_dir):
    # ładujemy kaskadę z pliku w resources/
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Nie można wczytać kaskady z {cascade_path}")

    for cls in classes:
        input_dir  = os.path.join(base_dir, cls)
        output_dir = os.path.join(input_dir, 'crops')
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            if len(faces) == 0:
                continue

            # wybieramy największą twarz
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            crop = img[y:y+h, x:x+w]
            crop = cv2.resize(crop, (200, 200))

            out_name = os.path.splitext(fname)[0] + '_crop.jpg'
            cv2.imwrite(os.path.join(output_dir, out_name), crop)
            print(f"[{base_dir}] {cls}/{fname} → crops/{out_name}")

if __name__ == "__main__":
    process_folder('resources/dataset')
    process_folder('resources/test')
