import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gc
import snowflake.connector
import h5py
import os
import gzip
from io import BytesIO

st.set_page_config(page_title="Détection de Voiture de Sport", page_icon="🚗")

# Fonction pour établir la connexion à Snowflake
def get_snowflake_connection():
    snowflake_secrets = st.secrets["snowflake"]
    return snowflake.connector.connect(
        user=snowflake_secrets["GROUPE2"],
        password=snowflake_secrets["P@ssword12345"],
        account=snowflake_secrets["DW10074"],
        warehouse=snowflake_secrets["COMPUTE_WH"],
        database=snowflake_secrets["TEST"],
        schema=snowflake_secrets["PUBLIC"],
    )

# Fonction pour télécharger et décompresser le modèle depuis Snowflake
def download_model_from_snowflake():
    conn = get_snowflake_connection()
    cursor = conn.cursor()

    try:
        # Téléchargement du fichier compressé depuis le stage Snowflake
        cursor.execute("GET @my_model_stage/final_sport_car_classifier.h5.gz file://./")
        local_compressed_file_path = "./final_sport_car_classifier.h5.gz"

        # Vérifier si le fichier a été téléchargé
        if os.path.exists(local_compressed_file_path):
            # Décompresser le fichier en mémoire
            with gzip.open(local_compressed_file_path, 'rb') as f:
                model_file = BytesIO(f.read())
            model = load_model(model_file)
            return model
        else:
            raise FileNotFoundError("Le fichier compressé n'a pas été téléchargé correctement.")
    finally:
        cursor.close()
        conn.close()

# Charger le modèle depuis Snowflake
try:
    model = download_model_from_snowflake()
    st.success("Modèle chargé avec succès.")
except FileNotFoundError as e:
    st.error("Erreur lors du téléchargement du modèle depuis Snowflake.")
    model = None  # Modèle non chargé
except Exception as e:
    st.error("Erreur inattendue lors du téléchargement du modèle.")
    model = None  # Modèle non chargé

# Configurer la mémoire de TensorFlow pour limiter l'utilisation
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Fonction de prétraitement de l'image
def preprocess_image(image):
    target_size = (224, 224, 3)
    image = image.resize(target_size[:2])
    image_array = np.array(image, dtype=np.float32) / 255.0
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Fonction de prédiction avec le modèle
def predict_with_model(image):
    if model is not None:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image, verbose=0)
        gc.collect()  # Libérer la mémoire après la prédiction
        return predictions[0][0]
    else:
        st.error("Le modèle n'a pas été chargé. Veuillez vérifier la connexion à Snowflake.")
        return None

# Interface utilisateur Streamlit

st.title("Détection de Voiture de Sport")
uploaded_file = st.file_uploader("Veuillez charger une image de voiture (jpg, jpeg, png)")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((image.width, 800))
    st.image(image, caption="Image téléchargée", use_column_width=True)

    if st.button("Predict"):
        sports_car_prob = predict_with_model(image)

        if sports_car_prob is not None:
            st.markdown(f"### Résultats de la prédiction :")
            st.write(f"- **Probabilité d'être une voiture de sport :** {sports_car_prob * 100:.2f}%")
            if sports_car_prob > 0.4:
                st.success("C'est probablement une voiture de sport ! 🚗")
            else:
                st.warning("Ce n'est probablement pas une voiture de sport.")
else:
    st.info("Veuillez charger une image pour activer la prédiction.")


# -------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# from PIL import Image
# import random
# import requests
# from io import BytesIO


# # Fonction pour charger l'image depuis une URL
# def load_image_from_url(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     return img


# # Configuration de la page
# st.set_page_config(
#     page_title="Voiture de Sport",
#     page_icon="🚗",
#     layout="centered",
#     initial_sidebar_state="collapsed",
# )

# # Styles CSS pour personnalisation
# st.markdown(
#     """
#     <style>
#         body {
#             background-color: #0e1117;
#             color: #ffffff;
#             font-family: 'Arial', sans-serif;
#         }
#         .header {
#             display: flex;
#             align-items: center;
#             justify-content: space-between;
#             background-color: #1e2228;
#             padding: 20px;
#             color: #ffffff;
#             font-weight: bold;
#             font-size: 22px;
#             border-radius: 8px;
#             margin-top: -70px;
#         }
#         .logo {
#             width: 50px;
#             margin-right: 20px;
#         }
#         .title {
#             font-size: 22px;
#         }
#         .subtitle {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 50px;
#         }
#         .upload-section {
#             text-align: center;
#             margin-top: 20px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Header avec le logo, le titre "Groupe 2" et le titre principal
# st.markdown(
#     """
#     <div class="header">
#         <img src="https://th.bing.com/th?id=OIP._KDC62yggz2WJ4qnK5jDfQHaEK&w=333&h=187&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2" alt="Logo" class="logo">
#         <div class="title">Détection de Voiture de Sport 🚗</div>
#         <div class="title">Groupe 2</div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # Upload de l'image
# uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])


# # Fonction de prédiction simulée avec pourcentages
# def predict_sports_car():
#     sports_car_score = random.uniform(0, 1)
#     not_sports_car_score = 1 - sports_car_score
#     return sports_car_score, not_sports_car_score


# # Affichage de l'image uploadée et prédiction
# if uploaded_file is not None:
#     # Charger l'image téléchargée
#     image = Image.open(uploaded_file)

#     # Redimensionner l'image téléchargée
#     image = image.resize((image.width, 800))

#     # Afficher l'image téléchargée redimensionnée
#     st.image(image, caption=" ", use_column_width=True)

#     # Bouton pour effectuer la prédiction
#     if st.button("Predict"):
#         # Obtenir les scores de prédiction
#         sports_car_score, not_sports_car_score = predict_sports_car()

#         # Afficher les résultats
#         st.markdown(f"### Résultats de la prédiction :")
#         st.write(
#             f"- **Probabilité d'être une voiture de sport :** {sports_car_score * 100:.2f}%"
#         )
#         st.write(
#             f"- **Probabilité de ne pas être une voiture de sport :** {not_sports_car_score * 100:.2f}%"
#         )

#         # Message de conclusion
#         if sports_car_score > not_sports_car_score:
#             st.success("C'est probablement une voiture de sport ! 🚗")
#         else:
#             st.warning("Ce n'est probablement pas une voiture de sport.")
# else:
#     # Message lorsque aucune image n'est téléchargée
#     st.info("Veuillez charger une image pour activer la prédiction.")