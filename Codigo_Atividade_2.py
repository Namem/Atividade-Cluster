import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import jaccard_score, fowlkes_mallows_score, adjusted_rand_score

def cluster_purity(true_labels, cluster_labels):
    purezas = []
    for cluster_id in np.unique(cluster_labels):
        members = true_labels[cluster_labels == cluster_id]
        if len(members) > 0:
            most_common = members.value_counts().max() / len(members)
            purezas.append(most_common)
    return purezas, np.mean(purezas)

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data.drop('label', axis=1)
    true_labels = data['label']
    return features, true_labels

def evaluate_clustering(features, true_labels, K_range):
    results = {k: {} for k in K_range}
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmlabels = kmeans.fit_predict(features)
        agglo = AgglomerativeClustering(n_clusters=k)
        agglabels = agglo.fit_predict(features)

        results[k]['km_rand'] = adjusted_rand_score(true_labels, kmlabels)
        results[k]['agg_rand'] = adjusted_rand_score(true_labels, agglabels)
        results[k]['km_jaccard'] = jaccard_score(true_labels, kmlabels, average='macro')
        results[k]['agg_jaccard'] = jaccard_score(true_labels, agglabels, average='macro')
        results[k]['km_fowlkes'] = fowlkes_mallows_score(true_labels, kmlabels)
        results[k]['agg_fowlkes'] = fowlkes_mallows_score(true_labels, agglabels)
        _, results[k]['km_pureza'] = cluster_purity(true_labels, kmlabels)
        _, results[k]['agg_pureza'] = cluster_purity(true_labels, agglabels)
    return results

def plot_metrics(results_df, K_range):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(K_range, results_df['km_rand'], 'o-', label="KMeans")
    plt.plot(K_range, results_df['agg_rand'], 'o-', label="Agglomerative")
    plt.title('Adjusted Rand Index')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(K_range, results_df['km_jaccard'], 'o-', label="KMeans")
    plt.plot(K_range, results_df['agg_jaccard'], 'o-', label="Agglomerative")
    plt.title('Jaccard Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(K_range, results_df['km_fowlkes'], 'o-', label="KMeans")
    plt.plot(K_range, results_df['agg_fowlkes'], 'o-', label="Agglomerative")
    plt.title('Fowlkes-Mallows Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(K_range, results_df['km_pureza'], 'o-', label="KMeans")
    plt.plot(K_range, results_df['agg_pureza'], 'o-', label="Agglomerative")
    plt.title('Pureza Média')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def print_best_k_results(results, features, true_labels, K_range):
    results_df = pd.DataFrame({
        'k': K_range,
        'km_rand': [results[k]['km_rand'] for k in K_range],
        'agg_rand': [results[k]['agg_rand'] for k in K_range],
        'km_jaccard': [results[k]['km_jaccard'] for k in K_range],
        'agg_jaccard': [results[k]['agg_jaccard'] for k in K_range],
        'km_fowlkes': [results[k]['km_fowlkes'] for k in K_range],
        'agg_fowlkes': [results[k]['agg_fowlkes'] for k in K_range],
        'km_pureza': [results[k]['km_pureza'] for k in K_range],
        'agg_pureza': [results[k]['agg_pureza'] for k in K_range],
    })

    # Para simplificar, seleciona o melhor k pelo Rand Agglomerative (alternativamente pelo Rand KMeans)
    idx_best_k_agg = results_df['agg_rand'].idxmax()
    best_k_agg = results_df['k'][idx_best_k_agg]
    idx_best_k_km = results_df['km_rand'].idxmax()
    best_k_km = results_df['k'][idx_best_k_km]

    agg = AgglomerativeClustering(n_clusters=best_k_agg)
    agg_labels = agg.fit_predict(features)
    kmeans = KMeans(n_clusters=best_k_km, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(features)

    print('\n---- Respostas Atividade 2 ----')

    print(f"1) Qual a quantidade de clusters?\nAgglomerative: {best_k_agg} | KMeans: {best_k_km}")

    print(f"\n2) Quantos pontos há em cada cluster?")
    print(f"Agglomerative:\n{pd.Series(agg_labels).value_counts().sort_index()}")
    print(f"KMeans:\n{pd.Series(km_labels).value_counts().sort_index()}")

    print("\n3) Qual a pureza dos clusters?")
    purezas_agglo, media_pureza_agglo = cluster_purity(true_labels, agg_labels)
    purezas_km, media_pureza_km = cluster_purity(true_labels, km_labels)
    for i, p in enumerate(purezas_agglo):
        print(f"Agglomerative - Cluster {i}: Pureza = {p:.3f}")
    print(f"Pureza média Agglomerative: {media_pureza_agglo:.3f}")

    for i, p in enumerate(purezas_km):
        print(f"KMeans - Cluster {i}: Pureza = {p:.3f}")
    print(f"Pureza média KMeans: {media_pureza_km:.3f}")

    print(f"\n4) Qual o coeficiente de Jaccard?")
    print(f"Agglomerative: {results_df['agg_jaccard'][idx_best_k_agg]:.3f}")
    print(f"KMeans: {results_df['km_jaccard'][idx_best_k_km]:.3f}")

    print(f"\n5) Qual o coeficiente de Rand?")
    print(f"Agglomerative: {results_df['agg_rand'][idx_best_k_agg]:.3f}")
    print(f"KMeans: {results_df['km_rand'][idx_best_k_km]:.3f}")

    print(f"\n6) Qual o coeficiente de Fowlkes Mallows?")
    print(f"Agglomerative: {results_df['agg_fowlkes'][idx_best_k_agg]:.3f}")
    print(f"KMeans: {results_df['km_fowlkes'][idx_best_k_km]:.3f}")

    print("\n7) Há diferença na performance dessas métricas se utilizar o KMeans ou o AgglomerativeClustering?")
    print(f"Rand Ajustado: Diferença = {abs(results_df['km_rand'][idx_best_k_km] - results_df['agg_rand'][idx_best_k_agg]):.3f}")
    print(f"Jaccard: Diferença = {abs(results_df['km_jaccard'][idx_best_k_km] - results_df['agg_jaccard'][idx_best_k_agg]):.3f}")
    print(f"Fowlkes-Mallows: Diferença = {abs(results_df['km_fowlkes'][idx_best_k_km] - results_df['agg_fowlkes'][idx_best_k_agg]):.3f}")
    print(f"Pureza média: Diferença = {abs(media_pureza_km - media_pureza_agglo):.3f}")

    print("\n8) Faça uma análise das características de cada grupo")
    print("Características dos grupos - Agglomerative:")
    print(features.groupby(agg_labels).mean())
    print("\nCaracterísticas dos grupos - KMeans:")
    print(features.groupby(km_labels).mean())

def main():
    features, true_labels = load_data('data_2.csv')
    K_range = range(2, 11)
    results = evaluate_clustering(features, true_labels, K_range)
    results_df = pd.DataFrame({
        'k': K_range,
        'km_rand': [results[k]['km_rand'] for k in K_range],
        'agg_rand': [results[k]['agg_rand'] for k in K_range],
        'km_jaccard': [results[k]['km_jaccard'] for k in K_range],
        'agg_jaccard': [results[k]['agg_jaccard'] for k in K_range],
        'km_fowlkes': [results[k]['km_fowlkes'] for k in K_range],
        'agg_fowlkes': [results[k]['agg_fowlkes'] for k in K_range],
        'km_pureza': [results[k]['km_pureza'] for k in K_range],
        'agg_pureza': [results[k]['agg_pureza'] for k in K_range],
    })
    plot_metrics(results_df, K_range)
    print_best_k_results(results, features, true_labels, K_range)

if __name__ == '__main__':
    main()
