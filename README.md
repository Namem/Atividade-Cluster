# 📊 Atividade - Análise Comparativa de Algoritmos de Clusterização

Este projeto realiza uma análise e comparação de dois algoritmos de clusterização, **KMeans** e **Aglomerativo Hierárquico**, aplicados a dois conjuntos de dados distintos.

O repositório está dividido em duas atividades principais:

1.  **Atividade 1 (`Codigo_Atividade_1.py`):** Foca na clusterização **não supervisionada**. O objetivo é encontrar o número ideal de *clusters* (k) e avaliar a qualidade dos grupos formados usando métricas internas (Elbow, Silhouette, Davies-Bouldin).
2.  **Atividade 2 (`Codigo_Atividade_2.py`):** Foca na avaliação **supervisionada** da clusterização. O objetivo é comparar o desempenho dos algoritmos com rótulos verdadeiros pré-existentes, usando métricas de avaliação externas (Pureza, Rand Ajustado, Jaccard, etc.).

## 📂 Estrutura do Projeto

```
.
├── .gitignore
├── Codigo_Atividade_1.py     # Script principal da Atividade 1
├── Codigo_Atividade_2.py     # Script principal da Atividade 2
├── data_1.csv                # Dados para a Atividade 1 (numérico)
├── data_2.csv                # Dados para a Atividade 2 (categórico/binário)
├── requirements.txt          # Dependências do projeto
└── ... (outros ficheiros de rascunho)
```

---

## 🚀 Atividade 1: Clusterização Não Supervisionada

Este script aplica o KMeans e o Clustering Aglomerativo ao `data_1.csv` para identificar agrupamentos naturais.

### Metodologia

1.  **Pré-processamento:** Os dados são carregados e normalizados usando `StandardScaler` para garantir que todas as *features* tenham a mesma escala.
2.  **Determinação de *k*:** Para encontrar o número ideal de *clusters* (`k`), o script testa valores de 2 a 10 e gera gráficos de avaliação para:
    * **Método do Cotovelo (Elbow Method):** Analisa a inércia (Soma dos Erros Quadráticos - SSE).
    * **Coeficiente de Silhouette:** Mede a separação e coesão dos *clusters* (idealmente próximo de 1).
    * **Índice de Davies-Bouldin:** Mede a similaridade entre os *clusters* (idealmente próximo de 0).
3.  **Clusterização Final:** Os algoritmos são executados com o `k` ideal (definido como 4 no script).
4.  **Visualização:** Os *clusters* resultantes de ambos os algoritmos são visualizados num gráfico de dispersão 2D, utilizando **PCA (Análise de Componentes Principais)** para reduzir a dimensionalidade dos dados.

---

## 🎯 Atividade 2: Avaliação Externa de Clusters

Este script utiliza o `data_2.csv`, um conjunto de dados categóricos (convertidos para *one-hot encoding*) que descreve perfis de clientes e já possui rótulos de classificação verdadeiros.

### Metodologia

1.  **Carregamento dos Dados:** Os dados são carregados, separando as *features* (ex: `idade19_29`, `sexo_masc`, `solteiro`) dos rótulos (`label`).
2.  **Avaliação Comparativa:** Os algoritmos KMeans e Aglomerativo são executados para valores de `k` de 2 a 10.
3.  **Métricas de Avaliação Externa:** A performance de cada `k` é medida comparando os rótulos previstos pelos algoritmos com os rótulos verdadeiros. As seguintes métricas são calculadas e plotadas:
    * **Pureza (Purity):** Mede a frequência da classe dominante em cada *cluster* (implementada numa função customizada).
    * **Adjusted Rand Score (ARI)**
    * **Jaccard Score**
    * **Fowlkes-Mallows Score**
4.  **Análise de Grupos:** Após identificar o melhor `k` (baseado no ARI), o script analisa as características de cada grupo, calculando a média das *features* para cada *cluster* formado. Isto permite criar uma "persona" ou descrição para cada segmento encontrado (ex: "Cluster 0 representa homens solteiros de 19-29 anos").

## 🛠️ Tecnologias Utilizadas

Este projeto utiliza as seguintes bibliotecas Python:

* **pandas:** Para manipulação e análise dos dados.
* **numpy:** Para operações numéricas.
* **scikit-learn:** Para os algoritmos de clusterização (KMeans, AgglomerativeClustering), pré-processamento (StandardScaler, PCA) e cálculo de métricas.
* **matplotlib:** Para a visualização dos gráficos.

## ⚡ Como Executar

**1. Clonar o repositório:**
```bash
git clone [https://github.com/Namem/Atividade-Cluster.git](https://github.com/Namem/Atividade-Cluster.git)
cd Atividade-Cluster
```

**2. (Recomendado) Criar um ambiente virtual:**
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

**3. Instalar as dependências:**
O ficheiro `requirements.txt` já está no repositório. Basta executar:
```bash
pip install -r requirements.txt
```

**4. Executar os scripts:**
```bash
# Para executar a primeira atividade
python Codigo_Atividade_1.py

# Para executar a segunda atividade
python Codigo_Atividade_2.py
```
