import io

import pandas as pd
import requests
import json


# Fonction pour télécharger les données à partir d'une URL
def request_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Erreur lors de la récupération des données depuis {url}")


# Fonction pour extraire un modèle à partir des données téléchargées
def extract_model(url):
    # Télécharger les données
    data = request_url(url)

    # Charger les données dans un DataFrame
    df = pd.read_csv(io.StringIO(data.decode('utf-8')))

    # Sélectionner les colonnes nécessaires pour le modèle
    selected_columns = ['Sex', 'Pclass', 'Age', 'Survived', 'Fare', 'Embarked']
    df = df[selected_columns]

    # Renommer les colonnes
    df = df.rename(columns={'Sex': 'sex', 'Pclass': 'class', 'Age': 'age', 'Survived': 'survived', 'Fare': 'price',
                            'Embarked': 'embarked'})

    return df


# Fonction pour nettoyer et formater les données
def transform(data):
    # Supprimer les lignes avec des valeurs manquantes
    data = data.dropna()

    # Créer une liste de dictionnaires représentant chaque passager
    passenger_list = data.to_dict(orient='records')

    return passenger_list


# Fonction pour enregistrer les données dans un fichier JSON
def load(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


# Utilisation des fonctions
if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    model_data = extract_model(data_url)
    transformed_data = transform(model_data)
    load(transformed_data, "passengers.json")
