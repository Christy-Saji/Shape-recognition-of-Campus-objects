import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# SETTINGS
# ======================
IMG_SIZE = 224
BATCH_SIZE = 16
DATASET_PATH = "dataset"
MODEL_PATH = "models/best_model.h5"

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# TEST GENERATOR
# ======================
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ======================
# EVALUATE
# ======================
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# ======================
# CONFUSION MATRIX
# ======================
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ======================
# SAVE METRICS TO JSON
# ======================
class_keys = list(test_generator.class_indices.keys())
report_dict = classification_report(y_true, y_pred, target_names=class_keys, output_dict=True)
cm_list = confusion_matrix(y_true, y_pred).tolist()

metrics_data = {
    "accuracy": round(accuracy * 100, 2),
    "confusion_matrix": cm_list,
    "classification_report": report_dict,
}

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

print("\nMetrics saved to metrics.json")