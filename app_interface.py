# ============================================================
#  Interfaz de Usuario — Clasificador de Enfermedades en Plantas
#  Ejecutar DESPUÉS de train_model.py
#  En Colab: simplemente corre este archivo y abre el link público
# ============================================================

# !pip install gradio tensorflow

import numpy as np
import json
import gradio as gr
import tensorflow as tf
from PIL import Image

# ── CONFIGURACIÓN ────────────────────────────────────────────
MODEL_PATH   = "plant_disease_model.keras"
CLASSES_PATH = "class_indices.json"
IMG_SIZE     = (224, 224)

# Descripciones amigables para algunas enfermedades comunes
DISEASE_INFO = {
    "healthy": {
        "status": "✅ Planta Sana",
        "advice": "Tu planta luce saludable. Mantén el riego regular y revisa periódicamente.",
        "color": "#27ae60"
    },
    "bacterial_spot": {
        "status": "⚠️ Mancha Bacteriana",
        "advice": "Aplica fungicida a base de cobre. Evita mojar las hojas al regar.",
        "color": "#e67e22"
    },
    "early_blight": {
        "status": "⚠️ Tizón Temprano",
        "advice": "Elimina hojas afectadas. Aplica fungicida. Mejora la ventilación.",
        "color": "#e67e22"
    },
    "late_blight": {
        "status": "🚨 Tizón Tardío",
        "advice": "Enfermedad severa. Aplica fungicida inmediatamente y aísla la planta.",
        "color": "#c0392b"
    },
    "leaf_mold": {
        "status": "⚠️ Moho de Hoja",
        "advice": "Reduce la humedad. Mejora la circulación de aire. Aplica fungicida.",
        "color": "#e67e22"
    },
    "mosaic_virus": {
        "status": "🚨 Virus del Mosaico",
        "advice": "No tiene cura. Elimina plantas afectadas para evitar propagación.",
        "color": "#c0392b"
    },
}

# ── CARGAR MODELO Y CLASES ────────────────────────────────────
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_indices = json.load(f)

# Invertir el diccionario: índice -> nombre de clase
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"Modelo cargado. {len(idx_to_class)} clases disponibles.")


# ── FUNCIÓN DE PREDICCIÓN ─────────────────────────────────────
def predict_disease(image: Image.Image):
    """
    Recibe una imagen PIL, la preprocesa y retorna
    la clase predicha con su probabilidad y top-3.
    """
    if image is None:
        return "Por favor sube una imagen.", "", ""

    # Preprocesar
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # Predecir
    predictions = model.predict(img_array, verbose=0)[0]
    top3_indices = np.argsort(predictions)[::-1][:3]

    # Resultado principal
    main_idx   = top3_indices[0]
    main_class = idx_to_class[main_idx]
    main_conf  = predictions[main_idx] * 100

    # Formatear nombre de clase para mostrar
    class_display = main_class.replace("___", " — ").replace("_", " ").title()

    # Buscar info de la enfermedad
    info_key = None
    for key in DISEASE_INFO:
        if key in main_class.lower():
            info_key = key
            break

    if info_key:
        status = DISEASE_INFO[info_key]["status"]
        advice = DISEASE_INFO[info_key]["advice"]
    else:
        status = "🔍 Enfermedad Detectada"
        advice = "Consulta con un agrónomo para un diagnóstico más detallado."

    # Top 3
    top3_text = "**Top 3 predicciones:**\n"
    for i, idx in enumerate(top3_indices):
        name = idx_to_class[idx].replace("___", " — ").replace("_", " ").title()
        conf = predictions[idx] * 100
        bar  = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
        top3_text += f"\n{i+1}. **{name}**\n   {bar} {conf:.1f}%\n"

    result = f"## {status}\n\n**Clase:** {class_display}\n\n**Confianza:** {main_conf:.1f}%"
    
    return result, advice, top3_text


# ── INTERFAZ GRADIO ───────────────────────────────────────────
with gr.Blocks(
    title="🌿 Plant Disease Classifier",
    theme=gr.themes.Soft(primary_hue="green")
) as demo:

    gr.Markdown("""
    # 🌿 Clasificador de Enfermedades en Plantas
    ### Powered by MobileNetV2 + Transfer Learning
    
    Sube una foto de una hoja y el sistema identificará si está **sana** o tiene alguna **enfermedad**.
    Entrenado con el dataset PlantVillage (54,000+ imágenes, 38 clases).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Sube una imagen de la hoja",
                height=300
            )
            predict_btn = gr.Button(
                "🔍 Analizar Planta",
                variant="primary",
                size="lg"
            )

            gr.Examples(
                examples=[],  # puedes agregar rutas a imágenes de ejemplo
                inputs=image_input,
                label="Ejemplos"
            )

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Resultado")
            advice_output = gr.Markdown(label="💡 Recomendación")
            top3_output   = gr.Markdown(label="Distribución de predicciones")

    predict_btn.click(
        fn=predict_disease,
        inputs=[image_input],
        outputs=[result_output, advice_output, top3_output]
    )

    gr.Markdown("""
    ---
    **Plantas soportadas:** Manzana, Arándano, Cereza, Maíz, Uva, Naranja, Durazno, 
    Pimiento, Papa, Frambuesa, Soya, Calabaza, Fresa, Tomate.
    
    *Proyecto Final — Inteligencia Artificial | UDLAP*
    """)

# ── LANZAR ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,        # genera link público en Colab
        debug=False,
        show_error=True
    )
