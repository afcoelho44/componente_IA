"""
modelo_dist.py — o que o treino e a API usam juntos.

O modelo prevê o ERRO da distância do GPS (erro_m = d_gps_m - distância da fita).
A distância corrigida é d_gps_m - erro_previsto. Prever o erro (e não a distância)
evita que o Random Forest "trave" na maior distância vista no treino: ele não
extrapola, mas o erro é pequeno e parecido em qualquer distância.
"""
import numpy as np
import pandas as pd

# ordem fixa: o modelo recebe sempre estas colunas, nesta ordem
FEATURES = ['d_gps_m', 'accuracy', 'outdoor', 'android', 'iphone']

# acima de FATOR_FAIXA × maior d_gps do treino, a API não corrige
FATOR_FAIXA = 1.5


def montar_X(df: pd.DataFrame, accuracy_padrao: float | None = None) -> pd.DataFrame:
    """Features a partir de d_gps_m, accuracy, ambiente e dispositivo.

    One-hot feito à mão (sem pd.get_dummies): com uma linha só, o get_dummies
    criaria só a coluna do valor presente e o modelo quebraria.
    """
    ambiente = df.get('ambiente', pd.Series('', index=df.index)).astype(str).str.lower()
    disp = df.get('dispositivo', df.get('dispositivo_curto', pd.Series('', index=df.index))).astype(str)

    X = pd.DataFrame(index=df.index)
    X['d_gps_m'] = pd.to_numeric(df['d_gps_m'], errors='coerce')
    X['accuracy'] = pd.to_numeric(df.get('accuracy', np.nan), errors='coerce')
    if accuracy_padrao is not None:
        X['accuracy'] = X['accuracy'].fillna(accuracy_padrao)
    X['outdoor'] = (ambiente == 'outdoor').astype(int)
    X['android'] = disp.str.contains('Android', case=False).astype(int)
    X['iphone'] = disp.str.contains('iPhone', case=False).astype(int)
    return X[FEATURES]


def distancia_corrigida(d_gps, erro_previsto):
    return np.maximum(0.0, np.asarray(d_gps, dtype=float) - np.asarray(erro_previsto, dtype=float))
