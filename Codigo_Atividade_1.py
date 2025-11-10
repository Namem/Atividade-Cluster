import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_normalize_data(filepath):
    df = pd.read_csv(filepath)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns), df

def clustering_and_metrics(df, n_clusters):
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(df)
    sil_kmeans = silhouette_score(df, labels_kmeans)
    db_kmeans = davies_bouldin_score(df, labels_kmeans)
    counts_kmeans = pd.Series(labels_kmeans).value_counts().sort_index()

    # Agglomerative
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agglo = agglo.fit_predict(df)
    sil_agglo = silhouette_score(df, labels_agglo)
    db_agglo = davies_bouldin_score(df, labels_agglo)
    counts_agglo = pd.Series(labels_agglo).value_counts().sort_index()

    return labels_kmeans, sil_kmeans, db_kmeans, counts_kmeans, labels_agglo, sil_agglo, db_agglo, counts_agglo

def plot_metrics(K_range, sse, silhouette_scores, db_scores):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(K_range, sse, marker='o')
    plt.title('SSE (Elbow Method)')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(K_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(K_range, db_scores, marker='o', color='red')
    plt.title('Davies-Bouldin Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_clusters(df, labels_kmeans, labels_agglo, n_clusters):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels_kmeans, cmap='viridis', s=50)
    plt.title(f'KMeans Clusters (PCA) - k={n_clusters}')
    plt.subplot(1, 2, 2)
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels_agglo, cmap='viridis', s=50)
    plt.title(f'Agglomerative Clusters (PCA) - k={n_clusters}')
    plt.tight_layout()
    plt.show()

def main():
    filepath = 'data_1.csv'
    df_scaled, df_original = load_and_normalize_data(filepath)
    K_range = range(2, 11)
    sse = []
    silhouette_scores = []
    db_scores = []
    # Avaliação de k para KMeans
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, labels))
        db_scores.append(davies_bouldin_score(df_scaled, labels))

    plot_metrics(K_range, sse, silhouette_scores, db_scores)

    # Responder questões para melhor k (Exemplo: k=4)
    best_k = 4
    result = clustering_and_metrics(df_scaled, best_k)
    labels_kmeans, sil_kmeans, db_kmeans, counts_kmeans, labels_agglo, sil_agglo, db_agglo, counts_agglo = result
    visualize_clusters(df_scaled, labels_kmeans, labels_agglo, best_k)

    print("\n---- Respostas Atividade 1 ----")
    print(f"1) Qual a quantidade de clusters?\nResposta: {best_k}\n")
    print(f"2) Quantos pontos há em cada cluster ?\n(KMeans)\n{counts_kmeans}\n(Agglomerative):\n{counts_agglo}\n")
    print(f"3) Qual foi o coeficiente de Silhouette ?\n(KMeans):{sil_kmeans:.3f}\n(Agglomerative):{sil_agglo:.3f}")
    print(f"4) Qual foi o coeficiente de Davies-Bouldin ?\n(KMeans):{db_kmeans:.3f}\n(Agglomerative):{db_agglo:.3f}")
    

    # Questão 5: Diferença de performance
    print("5) Há diferença na performance dessas métricas se utilizar o KMeans ou o AgglomerativeClustering?")
    print(f"Silhouette: KMeans={sil_kmeans:.3f}  Agglomerative={sil_agglo:.3f}")
    print(f"Davies-Bouldin: KMeans={db_kmeans:.3f}  Agglomerative={db_agglo:.3f}")
    if abs(sil_kmeans - sil_agglo) < 0.01 and abs(db_kmeans - db_agglo) < 0.01:
        print("Conclusão: Não houve diferença significativa entre os algoritmos neste conjunto de dados.")
    else:
        print("Conclusão: Houve diferença perceptível em algumas métricas.")

if __name__ == '__main__':
    main()
