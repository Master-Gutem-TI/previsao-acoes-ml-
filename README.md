# 🤖 Previsão de Preço de Ações com Machine Learning

Projeto de Machine Learning para prever o preço de fechamento de ações da B3,
comparando modelos de Regressão Linear e Random Forest com indicadores de análise técnica.

**Autor:** Gutemberg Nascimento de Souza  
**Formação:** MBA em Ciência de Dados para o Mercado Financeiro — XP Educação  
**Portfólio:** [github.com/Master-Gutem-TI](https://github.com/Master-Gutem-TI)

---

## Tecnologias utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![yfinance](https://img.shields.io/badge/yfinance-239120?style=flat)

---

## Sobre o projeto

Este projeto aplica técnicas de Machine Learning para prever o preço de fechamento
do dia seguinte de ações listadas na B3, usando dados históricos coletados
automaticamente via API do Yahoo Finance.

São criadas 11 features de análise técnica como variáveis preditoras, e dois modelos
são treinados e comparados usando métricas de regressão padrão do mercado.

---

## Features utilizadas (variáveis preditoras)

| Feature | Descrição |
|---|---|
| `mm_7` | Média móvel de 7 dias |
| `mm_21` | Média móvel de 21 dias |
| `mm_50` | Média móvel de 50 dias |
| `retorno_diario` | Variação percentual diária |
| `volatilidade_7d` | Desvio padrão dos retornos (7 dias) |
| `volatilidade_21d` | Desvio padrão dos retornos (21 dias) |
| `amplitude` | Amplitude do candle (máx - mín) / fechamento |
| `rsi` | Índice de Força Relativa (14 períodos) |
| `macd` | Moving Average Convergence Divergence |
| `macd_signal` | Linha de sinal do MACD (9 períodos) |
| `volume_change` | Variação percentual do volume |

---

## Modelos comparados

- **Regressão Linear** — modelo base de referência
- **Random Forest Regressor** — ensemble de árvores de decisão (100 estimadores)

### Métricas de avaliação

| Métrica | Descrição |
|---|---|
| MAE | Erro Médio Absoluto |
| RMSE | Raiz do Erro Quadrático Médio |
| R² | Coeficiente de determinação |

---

## Como executar

**Pré-requisitos:** Python 3.8+

```bash
# 1. Clone o repositório
git clone https://github.com/Master-Gutem-TI/previsao-acoes-ml.git
cd previsao-acoes-ml

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o projeto
python previsao_acoes.py
```

O script irá:
- Baixar automaticamente 2 anos de dados da ação `PETR4`
- Criar as 11 features de análise técnica
- Treinar e comparar os dois modelos
- Gerar o arquivo `previsao_acoes_ml.png` com os gráficos

---

## Estrutura do projeto

```
previsao-acoes-ml/
│
├── previsao_acoes.py        # Script principal
├── requirements.txt         # Dependências
├── previsao_acoes_ml.png    # Gráficos gerados (após execução)
└── README.md
```

---

## Resultados

Os gráficos gerados incluem:
- Previsão vs preço real (Random Forest)
- Previsão vs preço real (Regressão Linear)
- Importância das features (Random Forest)
- Scatter plot: real vs previsto (ambos os modelos)

---

## Contexto acadêmico

Este projeto foi desenvolvido como parte do portfólio prático do
**MBA em Ciência de Dados para o Mercado Financeiro** da XP Educação,
aplicando conceitos de:

- Engenharia de features com indicadores técnicos
- Treinamento e avaliação de modelos de regressão
- Comparação de algoritmos com métricas de negócio
- Análise de importância de variáveis

---

*Conecte-se comigo no [LinkedIn](https://linkedin.com/in/gutemberg-ti)*
