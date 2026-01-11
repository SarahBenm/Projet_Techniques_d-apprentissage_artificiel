import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplots(df, title_prefix="Boxplot de", color='lightcoral'):
    """
    Affiche les boxplots de toutes les colonnes numériques d'un DataFrame.
    
    Paramètres :
    -------------
    df : pandas.DataFrame
        Le DataFrame à analyser.
    title_prefix : str, optionnel
        Le texte affiché avant le nom de chaque colonne dans le titre du plot.
    color : str, optionnel
        Couleur des boxplots (par défaut : 'lightcoral').
    """
    # Sélection des colonnes numériques
    num_cols = df.select_dtypes(include='number').columns
    n_cols = len(num_cols)

    if n_cols == 0:
        print("Aucune colonne numérique trouvée dans le DataFrame.")
        return

    # Déterminer la disposition des sous-graphiques (2 par ligne)
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    axes = axes.flatten()

    # Générer les boxplots
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], color=color, ax=axes[i])
        axes[i].set_title(f"{title_prefix} {col}")
        axes[i].set_xlabel(col)

    # Supprimer les axes vides si nécessaire
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_distributions(df, bins=50,title_prefix="Distribution de", color='skyblue'):
    """
    Affiche les histogrammes avec KDE pour toutes les colonnes numériques d'un DataFrame.
    
    Paramètres :
    -------------
    df : pandas.DataFrame
        Le DataFrame à analyser.
    bins : int, optionnel
        Nombre de classes (par défaut : 50).
    color : str, optionnel
        Couleur des histogrammes (par défaut : 'skyblue').
    title_prefix : str, optionnel
        Préfixe du titre pour chaque histogramme.
    """
    # Colonnes numériques
    num_cols = df.select_dtypes(include='number').columns
    n_cols = len(num_cols)

    if n_cols == 0:
        print("Aucune colonne numérique trouvée dans le DataFrame.")
        return

    # Taille de la grille : 2 graphiques par ligne
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    axes = axes.flatten()

    # Tracer les histogrammes
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=bins, kde=True, color=color, ax=axes[i])
        axes[i].set_title(f"{title_prefix} {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Fréquence")

    # Supprimer les axes vides si le nombre de colonnes est impair
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()