# Previsão de Preço de Ações com Machine Learning
# Autor: Gutemberg Nascimento de Souza
# MBA em Ciência de Dados — XP Educação
# Ferramentas: Python, Scikit-learn, Pandas, Matplotlib, yfinance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. COLETA E PREPARAÇÃO DOS DADOS
# ============================================================

TICKER = 'PETR4.SA'
DIAS = 730  # 2 anos de dados

print(f"Baixando dados de {TICKER}...")
data_fim = datetime.today()
data_inicio = data_fim - timedelta(days=DIAS)
df = yf.download(TICKER, start=data_inicio, end=data_fim)[['Open','High','Low','Close','Volume']]
df.columns = ['abertura','maxima','minima','fechamento','volume']
df.dropna(inplace=True)
print(f"Registros: {len(df)} pregões\n")

# ============================================================
# 2. ENGENHARIA DE FEATURES (variáveis preditoras)
# ============================================================

# Médias móveis
df['mm_7']  = df['fechamento'].rolling(7).mean()
df['mm_21'] = df['fechamento'].rolling(21).mean()
df['mm_50'] = df['fechamento'].rolling(50).mean()

# Retorno diário e volatilidade
df['retorno_diario']    = df['fechamento'].pct_change()
df['volatilidade_7d']   = df['retorno_diario'].rolling(7).std()
df['volatilidade_21d']  = df['retorno_diario'].rolling(21).std()

# Amplitude do candle
df['amplitude'] = (df['maxima'] - df['minima']) / df['fechamento']

# RSI (Índice de Força Relativa)
delta = df['fechamento'].diff()
ganho = delta.where(delta > 0, 0).rolling(14).mean()
perda = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs    = ganho / perda
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
ema12 = df['fechamento'].ewm(span=12).mean()
ema26 = df['fechamento'].ewm(span=26).mean()
df['macd']        = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()

# Variação do volume
df['volume_change'] = df['volume'].pct_change()

# Alvo: preço de fechamento do dia seguinte
df['alvo'] = df['fechamento'].shift(-1)

df.dropna(inplace=True)

print("=== FEATURES CRIADAS ===")
features = ['mm_7','mm_21','mm_50','retorno_diario','volatilidade_7d',
            'volatilidade_21d','amplitude','rsi','macd','macd_signal','volume_change']
print(f"Total de features: {len(features)}")
for f in features:
    print(f"  - {f}")

# ============================================================
# 3. DIVISÃO TREINO / TESTE (80% / 20%)
# ============================================================

X = df[features]
y = df['alvo']

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Normalização
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTreino: {len(X_train)} registros | Teste: {len(X_test)} registros")

# ============================================================
# 4. TREINAMENTO DOS MODELOS
# ============================================================

modelos = {
    'Regressão Linear':    LinearRegression(),
    'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42),
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    pred = modelo.predict(X_test_sc)
    resultados[nome] = {
        'modelo':    modelo,
        'predicoes': pred,
        'MAE':  mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2':   r2_score(y_test, pred),
    }

# ============================================================
# 5. COMPARAÇÃO DE MODELOS
# ============================================================

print("\n=== COMPARAÇÃO DOS MODELOS ===")
print(f"{'Modelo':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)
for nome, res in resultados.items():
    print(f"{nome:<22} {res['MAE']:>8.4f} {res['RMSE']:>8.4f} {res['R2']:>8.4f}")

# Importância das features (Random Forest)
rf = resultados['Random Forest']['modelo']
importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n=== IMPORTÂNCIA DAS FEATURES (Random Forest) ===")
for feat, imp in importancias.items():
    barra = '█' * int(imp * 50)
    print(f"  {feat:<20} {barra} {imp:.4f}")

# ============================================================
# 6. VISUALIZAÇÕES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Previsão de Preço de Ações — {TICKER}', fontsize=14, fontweight='bold')
datas_teste = df.index[split:]

# Gráfico 1: Previsão vs Real (Random Forest)
ax1 = axes[0, 0]
ax1.plot(datas_teste, y_test.values, label='Real', color='#1f77b4', linewidth=1.5)
ax1.plot(datas_teste, resultados['Random Forest']['predicoes'],
         label='Random Forest', color='#ff7f0e', linewidth=1.2, linestyle='--')
ax1.set_title('Previsão vs Preço Real (Random Forest)')
ax1.set_ylabel('Preço (R$)')
ax1.legend()
ax1.tick_params(axis='x', rotation=30)

# Gráfico 2: Previsão vs Real (Regressão Linear)
ax2 = axes[0, 1]
ax2.plot(datas_teste, y_test.values, label='Real', color='#1f77b4', linewidth=1.5)
ax2.plot(datas_teste, resultados['Regressão Linear']['predicoes'],
         label='Regressão Linear', color='#2ca02c', linewidth=1.2, linestyle='--')
ax2.set_title('Previsão vs Preço Real (Regressão Linear)')
ax2.set_ylabel('Preço (R$)')
ax2.legend()
ax2.tick_params(axis='x', rotation=30)

# Gráfico 3: Importância das features
ax3 = axes[1, 0]
importancias.plot(kind='barh', ax=ax3, color='#1f77b4')
ax3.set_title('Importância das Features (Random Forest)')
ax3.set_xlabel('Importância')
ax3.invert_yaxis()

# Gráfico 4: Scatter Real vs Previsto
ax4 = axes[1, 1]
ax4.scatter(y_test, resultados['Random Forest']['predicoes'],
            alpha=0.5, s=15, color='#ff7f0e', label='Random Forest')
ax4.scatter(y_test, resultados['Regressão Linear']['predicoes'],
            alpha=0.5, s=15, color='#2ca02c', label='Regressão Linear')
min_val = min(y_test.min(), resultados['Random Forest']['predicoes'].min())
max_val = max(y_test.max(), resultados['Random Forest']['predicoes'].max())
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfeito')
ax4.set_xlabel('Preço Real (R$)')
ax4.set_ylabel('Preço Previsto (R$)')
ax4.set_title('Real vs Previsto')
ax4.legend()

plt.tight_layout()
plt.savefig('previsao_acoes_ml.png', dpi=150, bbox_inches='tight')
print("\nGráfico salvo: previsao_acoes_ml.png")
plt.show()
