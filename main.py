import urllib.request
import pandas as pd
import pipeline_csv_to_json


# Fonction pour charger les données du Titanic depuis une URL
def load_titanic_data(url):
    data = urllib.request.urlopen(url)  # Ouvrir l'URL et obtenir les données
    titanic_data = pd.read_csv(data)  # Lire les données CSV dans un DataFrame
    return titanic_data


# Fonction pour compter le nombre de femmes survivantes de moins de 18 ans
def count_female_survivors_under_18(titanic_data):
    mask_female = titanic_data['Sex'] == 'female'  # Créer un masque pour les femmes
    mask_under18 = titanic_data['Age'] < 18  # Créer un masque pour les personnes de moins de 18 ans
    female_under18_data = titanic_data[mask_female & mask_under18]  # Appliquer les deux masques
    female_under18_survived = sum(female_under18_data["Survived"])  # Compter les survivantes
    return female_under18_survived


# Fonction pour obtenir les données des femmes de moins de 18 ans
def count_female_under_18(titanic_data):
    mask_female = titanic_data['Sex'] == 'female'  # Créer un masque pour les femmes
    mask_under18 = titanic_data['Age'] < 18  # Créer un masque pour les personnes de moins de 18 ans
    female_under18_data = titanic_data[mask_female & mask_under18]  # Appliquer les deux masques
    return female_under18_data


# Fonction pour compter le nombre total de femmes à bord
def count_total_females(titanic_data):
    mask_female = titanic_data['Sex'] == 'female'  # Créer un masque pour les femmes
    nb_female = sum(mask_female)  # Compter le nombre de femmes en utilisant le masque
    return nb_female


# Fonction pour compter la répartition par classe
def count_class_distribution(titanic_data):
    class_distribution = titanic_data['Pclass'].value_counts()  # Compter les classes
    return class_distribution


# Fonction pour compter la répartition de la survie par port d'embarquement
def count_survival_by_port(titanic_data):
    grouped_by_embarked = titanic_data.groupby('Embarked')  # Regrouper par port d'embarquement
    # Compter le nombre de passagers ayant survécu (1) et n'ayant pas survécu (0) pour chaque port d'embarquement,
    # puis stocker les résultats dans le DataFrame survival_counts.
    # - 'grouped_by_embarked' : Les données du Titanic sont regroupées par port d'embarquement,
    #   créant un groupe de données distinct pour chaque port.
    # - ['Survived'] : À partir de chaque groupe, la colonne 'Survived' est sélectionnée, ce qui donne une série de données
    #   contenant les valeurs de survie (0 pour non survécu, 1 pour survécu) pour chaque passager de chaque port.
    # - .value_counts(): Compte le nombre d'occurrences de chaque valeur unique dans la série,
    #   fournissant ainsi le nombre de passagers ayant survécu et n'ayant pas survécu pour chaque port.
    # - .unstack() : Réorganise les données pour avoir le port d'embarquement comme index principal
    #   et les valeurs de survie (0 ou 1) comme colonnes. Chaque colonne correspond à une valeur de survie.
    # - .fillna(0) : Remplit les valeurs manquantes (NaN) par 0, garantissant ainsi que si un port n'a pas de passagers
    #   ayant une certaine valeur de survie, il est considéré comme ayant 0 passagers avec cette valeur de survie.
    survival_counts = grouped_by_embarked['Survived'].value_counts().unstack().fillna(0)
    return survival_counts


# Fonction pour calculer la répartition par âge en fonction du sexe
def calculate_age_distribution(titanic_data):
    # Regroupe les données par la colonne 'Sexe'
    grouped_by_sex = titanic_data.groupby('Sex')

    # Calculer la répartition des âges par sexe et stocker les résultats dans age_distribution.
    # - 'grouped_by_sex' : Les données du Titanic sont regroupées par sexe, créant un groupe de données distinct pour chaque sexe.
    # - ['Age'] : À partir de chaque groupe, la colonne 'Age' est sélectionnée, ce qui donne une série de données contenant
    #   les âges de chaque passager de chaque sexe.
    # - .value_counts() : Compte le nombre d'occurrences de chaque âge unique dans la série, fournissant ainsi la répartition
    #   des âges pour chaque sexe.
    # - .unstack() : Réorganise les données pour avoir le sexe en tant qu'index principal et les âges en colonnes.
    #   Chaque colonne correspond à un âge unique.
    # - .fillna(0) : Remplit les valeurs manquantes (NaN) par 0, garantissant ainsi que si un sexe n'a pas de passagers
    #   d'un certain âge, il est considéré comme ayant 0 passagers de cet âge.
    age_distribution = grouped_by_sex['Age'].value_counts().unstack().fillna(0)

    return age_distribution


# Fonction principale
def main():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    titanic_data = load_titanic_data(url)  # Charger les données du Titanic depuis l'URL

    female_under18_data = count_female_under_18(titanic_data)  # Obtenir les données des femmes de moins de 18 ans
    female_under18_survived = count_female_survivors_under_18(titanic_data)  # Compter les survivantes
    nb_female = count_total_females(titanic_data)  # Compter le nombre total de femmes
    class_distribution = count_class_distribution(female_under18_data)  # Compter la répartition par classe des femmes de moins de 18 ans
    count_survival_port = count_survival_by_port(titanic_data)  # Compter la répartition de la survie par port
    count_age_distribution = calculate_age_distribution(titanic_data)  # Compter la répartition par âge en fonction du sexe

    # Afficher les résultats
    print(f"Le nombre des femmes survivantes âgées de moins de 18 ans : {female_under18_survived}")
    print(f"Le nombre total de femmes à bord : {nb_female}")
    print(f"Répartition par classe parmi ces femmes: {class_distribution}")
    print(f"Répartition de la survie par port d'embarquement:\n{count_survival_port}")
    print(f"Répartition par sexe et par âge des passagers du navire:\n{count_age_distribution}")

    print(f"Data Ops pipeline started...")
    model_data = pipeline_csv_to_json.extract_model(url)
    print(f"Extracting data...")
    transformed_data = pipeline_csv_to_json.transform(model_data)
    print(f"Removing empty datas and transforming it to json...")
    pipeline_csv_to_json.load(transformed_data, "passengers.json")
    print(f"JSON file saved successfully passengers.json...")


if __name__ == "__main__":
    main()
