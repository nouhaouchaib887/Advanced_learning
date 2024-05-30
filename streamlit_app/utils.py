import pandas as pd
import numpy as np
import os
import streamlit as st 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import warnings
import plotly.express as px
import os
import plotly.figure_factory as ff
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')

#Parameters
train_dir0 = "/mount/src/Advanced_learning/streamlit_app/image/imd_wiki_examples"
train_examples = "/mount/src/Advanced_learning/streamlit_app/image/train_example"




SEED = 12
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢" , "😱" , "😊" , "😐 ", "😔" , "😲" ]
BATCH_SIZE = 64

#helpful functions


def display_multiple_images():
    image_paths = []
    titles = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', "Surprised", 'Angry', 'Disgusted', 'Fearful']
    
         
    for image_file in os.listdir(train_examples):
                image_path = os.path.join(train_examples, image_file)
                image_paths.append(image_path)
             

    


            
    fig, axes = plt.subplots(2,5, figsize=(5,5))
    axes = axes.flatten()
    for idx, (img_path, ax) in enumerate(zip(image_paths, axes)):
             img = cv2.imread(img_path)
             
             ax.imshow(img)
             ax.set_title(titles[idx])
             ax.axis('off')
             fig = plt.tight_layout()
    return fig

def plot_train_data_distribution():
    # Créer le graphique à barres
    fig = px.bar(
        x=CLASS_LABELS_EMOJIS,
        y = np.load('/mount/src/Advanced_learning/streamlit_app/resultats/data_distibution.npy'),
        color= np.array([0, 1, 2, 3, 4, 5, 6]),
        color_continuous_scale="Emrld"
    )
    
    # Mise à jour des axes et du layout
    fig.update_xaxes(title="Emotions")
    fig.update_yaxes(title="Number of Images")
    fig.update_layout(
        showlegend=True,
        title={
            'text': 'Train Data Distribution',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
 

    # Retourne la figure pour une utilisation éventuelle hors de la fonction
    return fig

def display_multiple_images_2():
    image_paths = []
         
   # Chemin complet vers le sous-dossier
    if os.path.isdir(train_dir0):  # Vérifier que c'est bien un dossier
        # Parcourir chaque image dans le sous-dossier
                for image_file in (os.listdir(train_dir0)) :
                    image_path = os.path.join(train_dir0, image_file)
                    image_paths.append(image_path)         
    fig, axes = plt.subplots(2,5, figsize=(5,5))
    axes = axes.flatten()
    for img_path, ax in zip(image_paths, axes):
             img = cv2.imread(img_path)
             img = cv2.resize(img, (48, 48))
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
             ax.imshow(img)
             ax.axis('off')
             fig = plt.tight_layout()
    return fig
def training_plots_accuracy(history):
     x = px.line(data_frame= history , y= ["accuracy" , "val_accuracy"] ,markers = True )
     x.update_xaxes(title="Number of Epochs")
     x.update_yaxes(title = "Accuracy")
     x.update_layout(showlegend = True,
    title = {
        'text': 'Accuracy vs Number of Epochs',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
     return x
def training_plots_loss(history):
     x = px.line(data_frame= history , 
            y= ["loss" , "val_loss"] , markers = True )
     x.update_xaxes(title="Number of Epochs")
     x.update_yaxes(title = "Loss")
     x.update_layout(showlegend = True,
    title = {
        'text': 'Loss vs Number of Epochs',
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
     return x
def evaluation_weighted(model_data ):
        metrics = { "Précision": "", "Rappel": "", "Score F1" :"", "Accuracy": ""}
        metrics["Précision"] = precision_score(model_data ["y_true"], model_data["y_pred"], average='weighted')
        metrics["Rappel"] =   recall_score(model_data ["y_true"], model_data["y_pred"], average='weighted')  
        metrics["Score F1"] = f1_score(model_data ["y_true"], model_data["y_pred"], average='weighted')
        metrics ["Accuracy" ] = accuracy_score(model_data ["y_true"], model_data["y_pred"])
        return metrics
def evaluation_macro(model_data ):
        metrics = { "Précision": "", "Rappel": "", "Score F1" :"", "Accuracy": ""}
        metrics["Précision"] = precision_score(model_data ["y_true"], model_data["y_pred"], average='macro')
        metrics["Rappel"] =   recall_score(model_data ["y_true"], model_data["y_pred"], average='macro')  
        metrics["Score F1"] = f1_score(model_data ["y_true"], model_data["y_pred"], average='macro')
        metrics ["Accuracy" ] = accuracy_score(model_data ["y_true"], model_data["y_pred"])
        return metrics
def matrice_confusion(data):
    cm = confusion_matrix(data["y_true"], data["y_pred"])

    fig = ff.create_annotated_heatmap(z=cm, x=CLASS_LABELS, y=CLASS_LABELS, colorscale='Blues', showscale=True)
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted labels', yaxis_title='True labels')

    return fig
def classification_rapport(data):
      y_true = data ["y_true"]
      y_pred = data["y_pred"]
      return classification_report(y_true, y_pred)
def report_to_df(report):
    lines = report.split('\n')
    rows = []
    for line in lines[2:-3]:  # Éviter les premières lignes de headers et les dernières lignes de résumé
        row = [value for value in line.split() if value]
        if row:
            rows.append(row)
    
    header = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']
    df = pd.DataFrame(rows, columns=header)
    index_last_row = df.index[-1]

# Supprimer la dernière ligne
    df = df.drop(index_last_row)
    return df


