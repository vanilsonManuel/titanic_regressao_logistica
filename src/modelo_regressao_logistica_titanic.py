# -*- coding: utf-8 -*-
"""


@author: Vanilson Mamuel
"""

"""
CASO PRÁTICO: Previsão de Sobrevivência no Titanic

Descrição:
Este script implementa um modelo de regressão logística utilizando o dataset
"titanic_v2.csv" para prever a sobrevivência dos passageiros.

Etapas:
1. Importação das bibliotecas
2. Carregamento dos dados
3. Pré-processamento
4. Treino do modelo
5. Avaliação do modelo
"""

# ==========================================
# 1️⃣ Importação das bibliotecas
# ==========================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from rich import print


# ==========================================
# 2️⃣ Carregamento do dataset
# ==========================================



def carregar_dados(caminho):
    """
    Carrega o dataset Titanic a partir de um ficheiro CSV.

    Parâmetros:
    caminho (str): caminho para o ficheiro CSV

    Retorna:
    DataFrame: dataset carregado
    """
    
    df = pd.read_csv(caminho)
    df.columns = df.columns.str.strip()
    return df


titanic = carregar_dados("../data/titanic_v2.csv")

print("Primeiras linhas:")
print(titanic.head())


# ==========================================
# 3️⃣ Pré-processamento
# ==========================================

def preprocessar_dados(df):
    """
    Realiza o pré-processamento dos dados:
    - Seleção de colunas relevantes
    - Remoção de valores nulos
    - Conversão de variáveis categóricas

    Retorna:
    DataFrame limpo e pronto para modelagem
    """

    colunas = ['survived', 'age', 'fare', 'pclass', 'sex']
    df = df[colunas]

    # Remover valores nulos
    df = df.dropna()

    # Converter categóricas em dummies
    df = pd.get_dummies(df, columns=['pclass', 'sex'], drop_first=True)

    return df


titanic = preprocessar_dados(titanic)

print("\nDataset após pré-processamento:")
print(titanic.head())


# ==========================================
# 4️⃣ Treino do modelo
# ==========================================

def treinar_modelo(df):
    """
    Treina um modelo de regressão logística.

    Retorna:
    modelo ajustado, dados de teste
    """

    X = df.drop('survived', axis=1)
    y = df['survived']

    # Converter para float (evita erro no statsmodels)
    X = X.astype(float)
    y = y.astype(float)

    # Usando NumPy (boa prática aqui)
    X = np.asarray(X)
    y = np.asarray(y)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Adicionar constante
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Criar e treinar modelo
    modelo = sm.Logit(y_train, X_train)
    resultado = modelo.fit()

    return resultado, X_test, y_test


resultado, X_test, y_test = treinar_modelo(titanic)

print("\nResumo do modelo:")
print(resultado.summary())


# ==========================================
# 5️⃣ Avaliação do modelo
# ==========================================

def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o modelo com matriz de confusão e accuracy.
    """

    # Previsões (probabilidades)
    y_prob = modelo.predict(X_test)

    # Usando NumPy para classificação
    y_pred = np.where(y_prob >= 0.5, 1, 0)

    # Matriz de confusão
    matriz = confusion_matrix(y_test, y_pred)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    return matriz, acc


matriz, acc = avaliar_modelo(resultado, X_test, y_test)

print("\nMatriz de Confusão:")
print(matriz)

print("\nTaxa de Acerto:", acc)
















