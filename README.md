# Projet Apprentissage Supervisé Avancé

## Auteurs
- Samy VILHES
- Ambre ADJEVI

## Instructions d'Installation

### 1. Cloner le Repository
```bash
git clone https://github.com/vilhess/methode-supervise.git
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement
Modifier les variables dans le fichier '.env' avec les valeurs appropriées.

### 4. Les Notebooks

Les notebooks où les codes sont renseignés se situent dans le dossier '/notebooks'
Afin d'éviter à avoir à re-entrainer les modèles CatBoost et réseaux neuronnaux, ces derniers seront sauvegardés dans le dossier '/models'.
Cependant, vous devez les entrainer une fois pour qu'ils soient sauvegardés (fichiers trop lourds pour être push sur GitHub)

### 5. Visualisation avec Streamlit

Il est possible de visualiser nos résulats sans avoir à run nos notebooks à l'aide de la commande à renseigner au niveau du terminal :

```bash
streamlit run page.py
```
