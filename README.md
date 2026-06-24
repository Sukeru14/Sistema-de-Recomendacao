# Sistemas de Recomendação — Trabalho Final

Implementação e comparação de algoritmos de filtragem colaborativa
(baseados em memória e em modelo) sobre o dataset MovieLens 100k.

## Estrutura

```
recsys/
├── data/ml-100k/         Dataset MovieLens 100k (943 usuários, 1682 itens, 100k notas)
│                         Inclui os 5 folds pré-definidos: u1.base/u1.test ... u5.base/u5.test
├── src/
│   ├── data_utils.py          Carregamento de dados e construção da matriz usuário-item
│   ├── memory_based.py        User-based KNN e Item-based KNN (implementados DO ZERO)
│   ├── model_based.py         Regularized SVD via SGD (implementado DO ZERO, acelerado com numba)
│   ├── evaluation.py          Métricas (RMSE, MAE) e infraestrutura de cross-validation
│   ├── hyperparam_search.py   Busca de hiperparâmetros um-de-cada-vez (holdout no fold u1)
│   ├── grid_search_svd.py     Busca em grade conjunta (n_factors x reg) para o SVD
│   ├── final_evaluation.py    Avaliação final via 5-fold CV com os melhores hiperparâmetros
│   ├── ranking_evaluation.py        EXTRA: métricas de qualidade da recomendação (Precision@K, Recall@K, NDCG@K, MAP@K)
│   ├── final_ranking_evaluation.py  EXTRA: avaliação final de ranking via 5-fold CV
│   ├── compare_surprise.py    EXTRA: comparação com SVD da biblioteca scikit-surprise
│   └── make_figures.py        Geração de todas as figuras a partir dos resultados salvos
├── results/               Resultados em JSON + logs de execução (.txt)
├── figures/                Gráficos gerados (.pdf)
└── requirements.txt
```

## O que foi implementado do zero vs. via biblioteca

| Algoritmo | Origem |
|---|---|
| User-based KNN (memória) | **Implementado do zero** (numpy) — `memory_based.py` |
| Item-based KNN (memória, extra) | **Implementado do zero** (numpy) — `memory_based.py` |
| Regularized SVD (modelo) | **Implementado do zero** (numpy + numba para acelerar o loop de SGD) — `model_based.py` |
| SVD via scikit-surprise (extra) | **Biblioteca externa**, usado apenas como validação/comparação — `compare_surprise.py` |

O numba é usado apenas para compilar (JIT) o mesmo loop de SGD que está escrito em Python puro
no docstring do módulo — não substitui a lógica do algoritmo por uma implementação de terceiros,
só acelera a execução (de ~20s para ~2s por fold).

## Como executar

```bash
pip install -r requirements.txt

cd src

# 1. Busca de hiperparâmetros um-de-cada-vez (holdout no fold u1) — gera results/hp_search_*.json
python3 hyperparam_search.py

# 2. Busca em grade conjunta (n_factors x reg) para o SVD — gera results/grid_search_svd.json
python3 grid_search_svd.py

# 3. Avaliação final via 5-fold CV com os melhores hiperparâmetros — gera results/final_results.json
python3 final_evaluation.py

# 4. (Extra) Comparação com scikit-surprise — gera results/svd_surprise_comparison.json
python3 compare_surprise.py

# 5. (Extra) Avaliação de qualidade de recomendação (ranking) — gera results/final_ranking_results.json
python3 final_ranking_evaluation.py

# 6. Gera todas as figuras a partir dos JSONs em results/
python3 make_figures.py
```