

# Análise Comparativa de Algoritmos de Clusterização

Este repositório contém os artefatos de um projeto acadêmico focado na análise e comparação de desempenho de algoritmos de clusterização (KMeans e Hierárquico Aglomerativo) em diferentes cenários de validação.

**Instituição:** Instituto Federal Do Mato Grosso (IFMT)-Campus Cuiabá
**Data:** 10 de Novembro de 2025

## 📜 Descrição do Projeto

O objetivo central deste trabalho foi conduzir uma análise comparativa entre o **KMeans** (um método particional) e a **Clusterização Hierárquica Aglomerativa** (um método hierárquico). O projeto foi dividido em duas atividades principais, conforme detalhado no relatório técnico.

### Atividade 1: Clusterização com Validação Interna

  * **Objetivo:** Aplicar os algoritmos em um conjunto de dados não rotulado (`data_1.csv`).
  * **Metodologia:**
      * Os dados foram pré-processados usando `StandardScaler` (normalização).
      * O número ótimo de clusters ($k=4$) foi determinado usando métricas de validação interna.
      * Métricas utilizadas: Coeficiente de Silhouette e Índice de Davies-Bouldin.
  * **Script:** `Codigo_Atividade_1.py`

### Atividade 2: Clusterização com Validação Externa

  * **Objetivo:** Avaliar os algoritmos em um conjunto de dados binário (`data_2.csv`) que possuía rótulos verdadeiros conhecidos.
  * **Metodologia:**
      * Não foi aplicada normalização devido à natureza binária dos dados.
      * O número ideal de clusters foi identificado como $k=3$.
      * Métricas utilizadas: Adjusted Rand Score, Jaccard Score e Pureza.
      * Foi realizada uma análise e interpretação dos perfis de grupo resultantes.
  * **Script:** `Codigo_Atividade_2.py`

## 📁 Estrutura do Repositório

```
/
├── Relatorio_Beatriz_Namem.pdf   # Relatório técnico completo do projeto
├── Codigo_Atividade_1.py         # Script Python para a Atividade 1 (Validação Interna)
├── Codigo_Atividade_2.py         # Script Python para a Atividade 2 (Validação Externa)
├── data_1.csv                    # Conjunto de dados para a Atividade 1 (não rotulado)
├── data_2.csv                    # Conjunto de dados para a Atividade 2 (rotulado)
├── requirements.txt              # Lista de dependências Python
├── Atividade1.ipynb              # Notebook Jupyter para a Atividade 1
├── Atividade2.ipynb              # Notebook Jupyter para a Atividade 2
└── .gitignore
```

## 🛠️ Tecnologias Utilizadas

As principais bibliotecas Python utilizadas neste projeto estão listadas no `requirements.txt` e incluem:

  * **pandas**
  * **numpy**
  * **scikit-learn** (para `KMeans`, `AgglomerativeClustering`, `StandardScaler` e métricas)
  * **matplotlib** (para visualização)
  * **seaborn**

## ⚙️ Configuração do Ambiente

Para executar este projeto, siga os passos abaixo:

1.  **Clone o repositório:**

    ```bash
    git clone <url-do-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Execução

Os scripts Python podem ser executados diretamente do terminal, desde que os arquivos `data_1.csv` e `data_2.csv` estejam no mesmo diretório.

### Executando a Atividade 1

Este script irá carregar `data_1.csv`, normalizar os dados, calcular as métricas de validação interna para $k$ de 2 a 10, exibir os gráficos de métricas (Cotovelo, Silhouette, Davies-Bouldin) e, por fim, imprimir os resultados detalhados para o $k=4$ ótimo.

```bash
python Codigo_Atividade_1.py
```

### Executando a Atividade 2

Este script irá carregar `data_2.csv`, calcular as métricas de validação externa (Rand, Jaccard, Pureza, etc.) para $k$ de 2 a 10, exibir os gráficos de métricas e, por fim, imprimir os resultados detalhados e a análise de perfil para o $k=3$ ótimo.

```bash
python Codigo_Atividade_2.py
```

## 🧑‍💻 Autores

  * Beatriz Aparecida Dutra Da Silva
  * Namem Rachid Jaudy Neto
  
-----