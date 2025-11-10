import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA





#Importação dos dados 
df = pd.read_csv('data_1.csv')


#visualizar a amostra de dados
print(df.head())
#print(df.describe())

#normatização dos dados 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled.head())

#Testantdo diferente Valores de k

sse = [] #Guarda o erro quadratico medio de cada clusterização, "inercia"
silhouette_scores= [] #Guarda o valor do coeficiente Shilhouette para cada k.
db_scores_kmeans = [] # Guarda o valor do Davies-Bouldin para cada k.
K = range(2,11) #Testa os valores de k 2 a 10 

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #Cria o modlo Kmens para o dado k
    kmeans.fit(df_scaled)#Executa a clustrização
    sse.append(kmeans.inertia_)#media da soma dos erros 
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))#Mede a separação entre cluster e a coerencia interna dos agrupamentos , quanto mais proximo de 1 melhor a separação
    db_scores_kmeans.append(davies_bouldin_score(df_scaled, kmeans.labels_))
    
# Plotando curva do cotovelo
'''
Mostra como o erro quadrático (inércia) diminui ao aumentar o número de clusters.
O ponto em que a curva “dobra” (fica menos inclinada) é considerado um bom candidato para k.
 '''
plt.plot(K, sse, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('SSE (inércia)')
plt.title('Método do Cotovelo')
plt.show()


print('\n****Respondendo as Questoes****')

# Rodar KMeans com k=4
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(df_scaled)

#plotar PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

# 1. Quantidade de clusters
num_clusters = len(set(labels_kmeans))
print('\nQuestão 1:')
print(f"Quantidade de clusters: {num_clusters}")

# 2. Quantos pontos há em cada cluster
points_per_cluster = pd.Series(labels_kmeans).value_counts().sort_index()
print('\nQuestão 2:')
print("Quantidade de pontos por cluster:")
print(points_per_cluster)

# 3. Silhouette Score
silhouette = silhouette_score(df_scaled, labels_kmeans)
print('\nQuestão 3:')
print(f"Coeficiente de Silhouette: {silhouette:.3f}")

# 4. Davies-Bouldin Index (quanto menor, melhor)
db_index = davies_bouldin_score(df_scaled, labels_kmeans)
print('\nQuestão 4:')
print(f"Coeficiente de Davies-Bouldin: {db_index:.3f}")

# 5. AgglomerativeClustering

silhouette_scores_agglo = []
db_scores_agglo = []
for k in range(2, 11):
    agglo = AgglomerativeClustering(n_clusters=k)
    labels = agglo.fit_predict(df_scaled)
    silhouette_scores_agglo.append(silhouette_score(df_scaled, labels))
    db_scores_agglo.append(davies_bouldin_score(df_scaled, labels))

agglo = AgglomerativeClustering(n_clusters=4)
labels_agglo = agglo.fit_predict(df_scaled)

# Gráfico comparativo do Silhouette
plt.figure(figsize=(10, 5))
plt.plot(K, silhouette_scores, '-o', label='KMeans')
plt.plot(K, silhouette_scores_agglo, '-o', label='Agglomerative')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette Score')
plt.title('Comparação do Silhouette Score')
plt.legend()
plt.grid(True)
plt.show()


# Gráfico comparativo do Davies-Bouldin
plt.figure(figsize=(10, 5))
plt.plot(K, db_scores_kmeans, '-o', label='KMeans')
plt.plot(K, db_scores_agglo, '-o', label='Agglomerative')
plt.xlabel('Número de clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Comparação do Davies-Bouldin Index')
plt.legend()
plt.grid(True)
plt.show()

# Visualização PCA lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(pca_data[:,0], pca_data[:,1], c=labels_kmeans, cmap='viridis', alpha=0.7)
axes[0].set_title('Visualização dos clusters KMeans (PCA)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[1].scatter(pca_data[:,0], pca_data[:,1], c=labels_agglo, cmap='viridis', alpha=0.7)
axes[1].set_title('Visualização dos clusters Agglomerative (PCA)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.show()

#Metricas Agglomerative

silhouette_agglo = silhouette_score(df_scaled, labels_agglo)
db_index_agglo = davies_bouldin_score(df_scaled, labels_agglo)
points_per_cluster_agglo = pd.Series(labels_agglo).value_counts().sort_index()
 
print('\nQuestão 5:')
print("Quantidade de pontos por cluster (Agglomerative):")
print(f"Silhouette Agglomerative: {silhouette_agglo:.3f} vs Silhouette KMeans: {silhouette:.3f}")
print(f"Davies-Bouldin Agglomerative: {db_index_agglo:.3f} vs Davies-Bouldin KMeans: {db_index:.3f}")
print("Pontos por cluster (Agglomerative):")
print(points_per_cluster_agglo)