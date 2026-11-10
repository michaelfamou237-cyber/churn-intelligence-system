
import pandas as pd
import numpy as np

def load_raw_data(filepath:str)->pd.DataFrame:
    return pd.read_csv(filepath)


# prise de connaissance du dataset 
def generate_data_audit(df:pd.DataFrame)->None : 
    print("-"*50)
    print("CONNAISSANCE DE NOTRE DATASET")
    print("-"*50)

    # dimension dataset
    print(f"\n---A. dimension--")
    print(f"nombre de lignes ou clients : {df.shape[0]}")
    print(f"nombre de variables : {df.shape[1]} \n")   

    # types de variabes 
    print(f"\n---B. types de variabes--")
    print(df.dtypes)

    # Modalités de chqaue variables
    print(f"\n---C. Nombre de modalités par variable --")
    print(df.nunique())

    # pourcentage de valeurs manquantes
    print(f"\n---C. Valeurs manquantes--")
    missing = df.isnull().sum()/ len(df)*100
    print(f"valeurs manquantes de chaque variable: {missing} \n")

    # Doublons:
    print(f"\n---D. Doublons --")
    duplicates = df.duplicated().sum()
    print(f"nombre de lignes dupliquées: {duplicates} \n")
    

    # Apercu statistiques des variables numeriques et categorielles
    print(f"\n---E. Statistiques de Bases-- ")
    print(f"\n-statistiques numeriques \n")
    var_num =  df.select_dtypes(exclude='O').describe().T
    print(var_num)
    
    print(f"\n-statistiques categorielles \n")
    var_cat =  df.select_dtypes(include='O').describe().T
    print(var_cat)
   