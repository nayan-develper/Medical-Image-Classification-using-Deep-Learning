import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "data/chest_xray/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = tf.keras.models.load_model("model.h5")

probs = model.predict(test_data).ravel()
preds = (probs > 0.5).astype(int)

y_true = test_data.classes

print("Accuracy:", accuracy_score(y_true, preds))
print("F1-score:", f1_score(y_true, preds))
print("ROC-AUC:", roc_auc_score(y_true, probs))
