
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Downloaded dataset location
DATA_DIR = (
    Path.home() / "Downloads" / "archive" / "dataset"
)

TRAIN_DIR = DATA_DIR / "TRAIN"
TEST_DIR = DATA_DIR / "TEST"

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        "Dataset folder not found. Check the archive/dataset path."
    )

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
CLASS_NAMES = ["O", "R"]

# Load training images and create validation set
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load separate test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Use pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Organic vs Recyclable model...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save under a NEW filename; preserve the existing model
model.save("binary_waste_model.keras")

with open("binary_class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file)

print("New binary model saved successfully!")
