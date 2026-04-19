# ============================================================
#  Clasificación de Enfermedades en Plantas — Entrenamiento
#  Ejecutar en Google Colab con GPU habilitada
#  Runtime > Change runtime type > T4 GPU
# ============================================================

# ── 1. INSTALAR DEPENDENCIAS (solo en Colab) ─────────────────
# !pip install tensorflow kaggle gradio matplotlib seaborn scikit-learn

# ── 2. DESCARGAR DATASET DESDE KAGGLE ───────────────────────
# Sube tu kaggle.json primero:
# from google.colab import files
# files.upload()   # sube kaggle.json
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d emmarex/plantdisease --unzip -p ./plantvillage

# ── 3. IMPORTS ───────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

# Fijar semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))


# ── 4. CONFIGURACIÓN GLOBAL ──────────────────────────────────
DATA_DIR    = "./plantvillage/PlantVillage"   # ajusta si la ruta cambia
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 10    # entrenar solo la cabeza nueva
EPOCHS_FINE = 10    # fine-tuning de las últimas capas
MODEL_PATH  = "plant_disease_model.keras"


# ── 5. PREPARACIÓN DE DATOS ──────────────────────────────────
# Aumentación solo en entrenamiento para mejorar generalización
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\nClases encontradas: {NUM_CLASSES}")
print(f"Imágenes de entrenamiento: {train_gen.samples}")
print(f"Imágenes de validación:    {val_gen.samples}")

# Guardar mapeo de clases para usarlo en la interfaz
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("class_indices.json guardado.")


# ── 6. CONSTRUCCIÓN DEL MODELO (Transfer Learning) ───────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    MobileNetV2 preentrenado en ImageNet como extractor de características.
    Se congela la base y se agrega una cabeza de clasificación nueva.
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # congelar pesos base

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(NUM_CLASSES)
model.summary()


# ── 7. FASE 1: ENTRENAR SOLO LA CABEZA ───────────────────────
callbacks_head = [
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
]

print("\n=== FASE 1: Entrenando cabeza de clasificación ===")
history_head = model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    callbacks=callbacks_head
)


# ── 8. FASE 2: FINE-TUNING (descongelar últimas capas) ───────
print("\n=== FASE 2: Fine-tuning de las últimas 30 capas ===")
base_model = model.layers[0]
base_model.trainable = True

# Congelar todo excepto las últimas 30 capas
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR bajo para fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_fine = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
]

history_fine = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    callbacks=callbacks_fine
)


# ── 9. EVALUACIÓN ────────────────────────────────────────────
def plot_history(h1, h2, save=True):
    """Grafica accuracy y loss de ambas fases de entrenamiento."""
    acc  = h1.history['accuracy']      + h2.history['accuracy']
    val  = h1.history['val_accuracy']  + h2.history['val_accuracy']
    loss = h1.history['loss']          + h2.history['loss']
    vloss= h1.history['val_loss']      + h2.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, acc,  label='Train Accuracy',  color='#2ecc71')
    axes[0].plot(epochs, val,  label='Val Accuracy',    color='#e74c3c', linestyle='--')
    axes[0].axvline(x=len(h1.history['accuracy']), color='gray', linestyle=':', label='Fine-tune start')
    axes[0].set_title('Accuracy por época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, loss,  label='Train Loss',  color='#2ecc71')
    axes[1].plot(epochs, vloss, label='Val Loss',    color='#e74c3c', linestyle='--')
    axes[1].axvline(x=len(h1.history['loss']), color='gray', linestyle=':', label='Fine-tune start')
    axes[1].set_title('Loss por época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("training_curves.png", dpi=150)
        print("Gráfica guardada: training_curves.png")
    plt.show()

plot_history(history_head, history_fine)

# Cargar el mejor modelo guardado
best_model = tf.keras.models.load_model(MODEL_PATH)

# Evaluar en validación
print("\n=== Evaluación final en datos de validación ===")
loss, acc = best_model.evaluate(val_gen, verbose=1)
print(f"Accuracy final: {acc*100:.2f}%")
print(f"Loss final:     {loss:.4f}")

# Reporte por clase y matriz de confusión
print("\n=== Generando reporte de clasificación ===")
val_gen.reset()
y_pred_probs = best_model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

class_names = list(train_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusión (top 10 clases para visualización)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm, annot=False, fmt='d', cmap='YlOrRd',
    xticklabels=class_names, yticklabels=class_names, ax=ax
)
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title('Matriz de Confusión', fontsize=14)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Matriz guardada: confusion_matrix.png")
plt.show()

print("\n✅ Entrenamiento completado.")
print(f"   Modelo guardado en: {MODEL_PATH}")
