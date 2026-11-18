"""
treinar.py — treina o modelo que corrige a distância do GPS.

Uso:
    python treinar.py dados/erro_distancia_por_leitura.csv modelos/modelo.joblib

Entrada: erro_distancia_por_leitura.csv do analisar.py (experimento com fita).
Só as linhas do fluxo 'bruto' entram, porque o app manda a distância do GPS sem Kalman.

Imprime a validação por participante (GroupKFold) comparando três correções:
  - GPS bruto      : sem correção
  - viés médio     : d_gps - média do erro no treino
  - Random Forest  : d_gps - erro previsto
"""
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from modelo_dist import FEATURES, distancia_corrigida, montar_X


def novo_modelo():
    # min_samples_leaf alto: leituras da mesma captura são quase iguais e o modelo decoraria
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1)


def carregar(caminho):
    df = pd.read_csv(caminho)
    if 'fluxo' in df.columns:
        df = df[df['fluxo'] == 'bruto']
    if 'dispositivo' not in df.columns and 'dispositivo_curto' in df.columns:
        df = df.rename(columns={'dispositivo_curto': 'dispositivo'})
    if 'erro_m' not in df.columns:
        df['erro_m'] = df['d_gps_m'] - df['distancia_m']
    df = df.dropna(subset=['d_gps_m', 'distancia_m', 'erro_m']).reset_index(drop=True)
    df['participante'] = df.get('participante', 'sem_id').astype(str)
    return df


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def validar(df, X, y):
    grupos = df['participante']
    n = grupos.nunique()
    if n < 2:
        print(f'\nSó {n} participante: sem validação cruzada (precisa de 2 ou mais).')
        return None

    linhas = []
    for treino, teste in GroupKFold(n_splits=min(5, n)).split(X, y, grupos):
        acc_med = X.iloc[treino]['accuracy'].median()
        Xtr, Xte = X.iloc[treino].fillna({'accuracy': acc_med}), X.iloc[teste].fillna({'accuracy': acc_med})
        rf = novo_modelo().fit(Xtr, y.iloc[treino])

        fita = df['distancia_m'].iloc[teste].to_numpy()
        d_gps = df['d_gps_m'].iloc[teste].to_numpy()
        correcoes = {
            'GPS bruto': d_gps,
            'viés médio': distancia_corrigida(d_gps, y.iloc[treino].mean()),
            'Random Forest': distancia_corrigida(d_gps, rf.predict(Xte)),
        }
        for nome, dist in correcoes.items():
            for amb in df['ambiente'].iloc[teste].unique() if 'ambiente' in df else ['todos']:
                m = (df['ambiente'].iloc[teste] == amb).to_numpy() if 'ambiente' in df else slice(None)
                linhas.append({'metodo': nome, 'ambiente': amb, 'erro': dist[m] - fita[m]})

    res = []
    for (metodo, amb), g in pd.DataFrame(linhas).groupby(['metodo', 'ambiente'], sort=False):
        e = np.concatenate(g['erro'].to_numpy())
        res.append({'ambiente': amb, 'metodo': metodo, 'n': len(e),
                    'vies_m': e.mean(), 'ME_m': np.abs(e).mean(), 'RMSE_m': rmse(e)})
    tabela = pd.DataFrame(res).sort_values(['ambiente', 'RMSE_m'])
    print(f'\n== Validação por participante (GroupKFold, {n} participantes) ==')
    print(tabela.round(2).to_string(index=False))
    return tabela


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('--tabela', default=None, help='salva a tabela da validação em CSV')
    a = ap.parse_args()

    df = carregar(a.entrada)
    X = montar_X(df)
    y = df['erro_m']

    print(f'{len(df)} leituras (fluxo bruto), {df["participante"].nunique()} participantes')
    if 'ambiente' in df:
        print(pd.crosstab(df['ambiente'], df['distancia_m']).to_string())

    tabela = validar(df, X, y)
    if tabela is not None and a.tabela:
        tabela.to_csv(a.tabela, index=False)

    acc_med = float(X['accuracy'].median()) if X['accuracy'].notna().any() else 5.0
    modelo = novo_modelo().fit(X.fillna({'accuracy': acc_med}), y)

    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n== Importância das features ==')
    print(imp.round(3).to_string())

    joblib.dump({
        'modelo': modelo,
        'features': FEATURES,
        'sklearn': sklearn.__version__,
        'treinado_em': datetime.now().isoformat(timespec='seconds'),
        'participantes': int(df['participante'].nunique()),
        'leituras': int(len(df)),
        'accuracy_mediana': acc_med,
        'faixa_d_gps': [float(df['d_gps_m'].min()), float(df['d_gps_m'].max())],
    }, a.saida)
    print(f'\nModelo salvo em {a.saida}')


if __name__ == '__main__':
    main()
