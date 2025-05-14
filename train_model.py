import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

classes = ['messi', 'other']

face_train_dir = 'resources/dataset_faces'
face_val_dir   = 'resources/test_faces'

# Usuń jeśli już istniały, i stwórz na nowo
for d in (face_train_dir, face_val_dir):
    if os.path.exists(d):
        shutil.rmtree(d)
    for cls in classes:
        os.makedirs(os.path.join(d, cls), exist_ok=True)

# Skopiuj crop’y do nowych katalogów
for cls in classes:
    src_train = os.path.join('resources', 'dataset', cls, 'crops')
    dst_train = os.path.join(face_train_dir, cls)
    if os.path.isdir(src_train):
        for f in os.listdir(src_train):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                shutil.copy(os.path.join(src_train, f), os.path.join(dst_train, f))

    src_val = os.path.join('resources', 'test', cls, 'crops')
    dst_val = os.path.join(face_val_dir, cls)
    if os.path.isdir(src_val):
        for f in os.listdir(src_val):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                shutil.copy(os.path.join(src_val, f), os.path.join(dst_val, f))


# ==============================================================
# 1) Generatory z augmentacją dla train i bez dla walidacji
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    face_train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    face_val_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)


# ==============================================================
# 2) Definicja modelu CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')   # wyjście w [0,1]: p(other)
])


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = 15
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)


loss, acc = model.evaluate(val_gen)
print(f"\nTest loss: {loss:.4f},  Test accuracy: {acc:.4f}")

model.save('messi_binary_classifier.h5')
print("Model zapisany jako messi_binary_classifier.h5")
