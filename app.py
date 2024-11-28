
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gc
import snowflake.connector
import h5py
import os

import streamlit as st
st.set_page_config(page_title="Détection de Voiture de Sport", page_icon="\U0001F697")


def get_snowflake_connection():
    snowflake_secrets = st.secrets["snowflake"]
    return snowflake.connector.connect(
        user=snowflake_secrets["GROUPE2"],
        warehouse =snowflake_secrets["COMPUTE_WH"] 
        password=snowflake_secrets["P@ssword12345"],
        account=snowflake_secrets["DW10074"],
    )

# Fonction de connexion et de téléchargement du modèle depuis Snowflake
def download_model_from_snowflake():
    # Connexion à Snowflake
    conn = snowflake.connector.connect(
        user="GROUPE2",
        password="P@ssword12345",
        account="DW10074",
        warehouse="YOUR_WAREHOUSE",  # Précisez votre entrepôt de données
        database="TEST",  # Précisez votre base de données
        schema="PUBLIC",  # Précisez votre schéma
    )

    cursor = conn.cursor()

    try:
        # Téléchargement du fichier depuis le stage Snowflake
        cursor.execute("GET @MY_MODEL_STAGE/final_sport_car_classifier.h5 file://./")
        local_file_path = "./final_sport_car_classifier.h5"

        # Vérifier si le fichier a été téléchargé
        if os.path.exists(local_file_path):
            model = h5py.File(local_file_path, "r")
            return model
        else:
            raise FileNotFoundError("Le fichier n'a pas été téléchargé correctement.")
    finally:
        cursor.close()
        conn.close()


# Charger le modèle depuis Snowflake
try:
    model_file = download_model_from_snowflake()
    model = load_model(model_file)
    print("Modèle chargé avec succès.")
except FileNotFoundError as e:
    print(e)
    st.error(
        "Erreur lors du téléchargement du modèle depuis Snowflake. Veuillez vérifier la connexion et le chemin du fichier."
    )
    model = None  # Modèle non chargé
except Exception as e:
    print(f"Erreur : {e}")
    st.error("Une erreur inattendue est survenue lors du téléchargement du modèle.")
    model = None  # Modèle non chargé


# Configurer la mémoire de TensorFlow pour limiter l'utilisation
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def preprocess_image(image):
    """Prétraite l'image pour qu'elle corresponde au format attendu par le modèle."""
    target_size = (224, 224, 3)  # Taille plus petite pour économiser de la mémoire
    image = image.resize(target_size[:2])
    image_array = np.array(image, dtype=np.float32) / 255.0
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_with_model(image):
    """Prédit avec le modèle si l'image contient une voiture de sport."""
    if model is not None:
        preprocessed_image = preprocess_image(image)
        additional_input = np.zeros((1, 1335), dtype=np.float32)
        predictions = model.predict([preprocessed_image, additional_input], verbose=0)
        gc.collect()  # Libérer de la mémoire après la prédiction
        return predictions[0][0]
    else:
        st.error(
            "Le modèle n'a pas été chargé. Veuillez vérifier la connexion à Snowflake."
        )
        return None


# Interface utilisateur Streamlit
st.title("Détection de Voiture de Sport")
uploaded_file = st.file_uploader(
    "Veuillez charger une image de voiture (jpg, jpeg, png)"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    if st.button("Predict"):
        sports_car_prob = predict_with_model(image)

        if sports_car_prob is not None:
            if sports_car_prob > 0.4:
                st.success(
                    f"✔ C'est probablement une voiture de sport  ! (Score: {sports_car_prob * 100:.2f}%)"
                )
            else:
                st.error(
                    f"✘ Ce n'est probablement pas une voiture de sport. (Score: {sports_car_prob * 100:.2f}%)"
                )
else:
    st.info("Veuillez charger une image pour activer la prédiction.")
