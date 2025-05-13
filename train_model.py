import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ścieżki do danych
train_dir = 'resources/dataset'
val_dir   = 'resources/test'

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
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',        # bo mamy 2 klasy
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

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
    Dense(1, activation='sigmoid')   # wyjście w [0,1]: p(Messi)
])

# 3) Kompilacja
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4) Trening
epochs = 15
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# 5) Ocena na zbiorze testowym
loss, acc = model.evaluate(val_gen)
print(f"\nTest loss: {loss:.4f},  Test accuracy: {acc:.4f}")

# 6) Zapis modelu
model.save('messi_binary_classifier.h5')
print("Model zapisany jako messi_binary_classifier.h5")
 