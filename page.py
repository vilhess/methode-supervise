import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown(f"<h1 style='text-align: center;'>Projet Supervisé Avancé</h1>", unsafe_allow_html=True)
st.subheader('')
st.subheader('')

probleme = st.sidebar.selectbox(
    'Probleme : ', ['Classification', 'Régression'])

if probleme == 'Régression':

    data_train_reg_path = 'data/data_reg/wine_train.csv'

    df_train_reg = pd.read_csv(data_train_reg_path).drop('wine_ID', axis=1)
    st.table(df_train_reg.sample(5))

    st.markdown(f"<h2 style='text-align: center;'>Description du jeu de données</h2>", unsafe_allow_html=True)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Format du jeu de données : {df_train_reg.shape[0]} lignes et {df_train_reg.shape[1]} variables (dont celle à prédire) </h5>", unsafe_allow_html=True)

    st.subheader('')
    
    st.table(df_train_reg.describe())

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Corrélation entre les variables</h5>", unsafe_allow_html=True)

    st.subheader('')

    corr = df_train_reg.corr()

    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True)
    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Distribution des variables</h5>", unsafe_allow_html=True)

    st.subheader('')

    fig = plt.figure(figsize=(15, 15))

    for i in range(df_train_reg.shape[1]):
        fig.add_subplot(4, 4, i+1)
        sns.histplot(df_train_reg.iloc[:, i], color='green', label=df_train_reg.columns[i])
        plt.axvline(df_train_reg.iloc[:, i].mean(), linestyle='dashed', color='red', label='mean')
        plt.axvline(df_train_reg.iloc[:, i].median(), linestyle='dashed', color='blue', label='median')
        plt.legend()

    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Analyse en composantes principales</h5>", unsafe_allow_html=True)

    st.subheader('')

    pca = PCA()
    pca.fit(df_train_reg.drop('target', axis=1))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_.cumsum())
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Pourcentage de variance expliqué cumulé')
    plt.grid()
    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Plans factoriels</h5>", unsafe_allow_html=True)

    st.subheader('')

    st.pyplot(fig)

    plan_principaux = st.sidebar.slider('Plan factoriel à afficher', 1, 10, 1)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Projection des individus sur le plan factoriel {plan_principaux}</h5>", unsafe_allow_html=True)

    st.subheader('')

    fig = plt.figure(figsize=(10, 10))

    new_df = pd.DataFrame(pca.transform(df_train_reg.drop('target', axis=1)), columns=[
                            'PC' + str(i) for i in range(1, df_train_reg.shape[1])])

    sns.scatterplot(x='PC' + str(plan_principaux), y='PC' + str(plan_principaux + 1), data=new_df, hue=df_train_reg['target'], palette='Set3')
    plt.xlabel('PC' + str(plan_principaux))
    plt.ylabel('PC' + str(plan_principaux + 1))
    plt.grid()
    st.pyplot(fig)

else :

    data_train_cla_path = 'data/data_cla/stars_train.csv'

    df_train_cla = pd.read_csv(data_train_cla_path).drop('obj_ID', axis=1)
    st.table(df_train_cla.sample(5))

    st.markdown(f"<h2 style='text-align: center;'>Description du jeu de données</h2>", unsafe_allow_html=True)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Format du jeu de données : {df_train_cla.shape[0]} lignes et {df_train_cla.shape[1]} variables (dont celle à prédire) </h5>", unsafe_allow_html=True)

    st.subheader('')

    st.table(df_train_cla.describe())

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Corrélation entre les variables</h5>", unsafe_allow_html=True)
    
    st.subheader('')

    corr = df_train_cla.corr()

    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True)
    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Distribution des variables</h5>", unsafe_allow_html=True)

    st.subheader('')

    fig = plt.figure(figsize=(15, 15))

    for i in range(df_train_cla.shape[1]):
        fig.add_subplot(4, 4, i+1)
        sns.histplot(df_train_cla.iloc[:, i], color='green', label=df_train_cla.columns[i])
        plt.axvline(df_train_cla.iloc[:, i].mean(), linestyle='dashed', color='red', label='mean')
        plt.axvline(df_train_cla.iloc[:, i].median(), linestyle='dashed', color='blue', label='median')
        plt.legend()

    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Analyse en composantes principales</h5>", unsafe_allow_html=True)

    st.subheader('')

    pca = PCA()
    pca.fit(df_train_cla.drop('label', axis=1))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_.cumsum())
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Pourcentage de variance expliqué cumulé')
    plt.grid()
    st.pyplot(fig)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Plans factoriels</h5>", unsafe_allow_html=True)

    st.subheader('')

    plan_principaux = st.sidebar.slider('Plan factoriel à afficher', 1, 10, 1)

    st.subheader('')

    st.markdown(f"<h5 style='text-align: center;'>Projection des individus sur le plan factoriel {plan_principaux}</h5>", unsafe_allow_html=True)

    st.subheader('')

    fig = plt.figure(figsize=(10, 10))

    new_df = pd.DataFrame(pca.transform(df_train_cla.drop('label', axis=1)), columns=[
                            'PC' + str(i) for i in range(1, df_train_cla.shape[1])])
    
    sns.scatterplot(x='PC' + str(plan_principaux), y='PC' + str(plan_principaux + 1), data=new_df, hue=df_train_cla['label'], palette='Set3')
    plt.xlabel('PC' + str(plan_principaux))
    plt.ylabel('PC' + str(plan_principaux + 1))
    plt.grid()
    st.pyplot(fig)

    