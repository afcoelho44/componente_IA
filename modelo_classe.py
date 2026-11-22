"""
modelo_classe.py — o que o treino e a API usam juntos.

O componente classifica a proximidade do aluno ao objeto em três faixas:
    quente : distância <= 10 m
    morno  : 10 m < distância <= 15 m
    frio   : distância > 15 m

No treino, a classe "verdadeira" vem da distância medida com fita; as features
vêm só das leituras do GPS (presente e passado), as mesmas que a API calcula ao vivo.
A "regra" é a abordagem determinística: os mesmos limites aplicados direto na
distância do GPS. O trabalho compara as duas.
"""
import numpy as np
import pandas as pd

# limites superiores (m) de cada faixa, em ordem; acima do último é frio
LIMITES = [(10.0, 'quente'), (15.0, 'morno')]
CLASSES = ['quente', 'morno', 'frio']

JANELA = 5           # leituras usadas nas features de histórico (mediana, desvio)
R_TERRA = 6371008.8

# ordem fixa: o modelo recebe sempre estas colunas, nesta ordem
FEATURES = [
    'dist',          # distância do GPS até o objeto na leitura atual (m)
    'delta_dist',    # variação da distância desde a leitura anterior (m)
    'dist_med',      # mediana da distância nas últimas JANELA leituras (m)
    'dist_desv',     # desvio-padrão da distância nas últimas JANELA leituras (m)
    'accuracy',      # precisão horizontal informada pelo navegador (m)
    'speed',         # velocidade do GPS (m/s, 0 quando o navegador não informa)
    'iphone',        # 1 = iPhone, 0 = outro
]


def classe_por_distancia(dist):
    """Aplica os limites. Aceita número, lista, array ou Series."""
    d = np.asarray(dist, dtype=float)
    saida = np.select([d <= lim for lim, _ in LIMITES], [nome for _, nome in LIMITES], default='frio')
    return saida.item() if saida.ndim == 0 else saida


def haversine(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lng2, dtype=float) - np.asarray(lng1, dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TERRA * np.arcsin(np.sqrt(a))


def preparar(df: pd.DataFrame, grupo: str = 'captura_id') -> pd.DataFrame:
    """Calcula as FEATURES para cada leitura.

    Colunas esperadas: grupo (sequência contínua de leituras), t (segundos, para
    ordenar), lat, lng, obj_lat, obj_lng, accuracy; opcionais: speed, dispositivo.
    Só usa a leitura atual e as anteriores do mesmo grupo — nunca o futuro.
    """
    df = df.sort_values([grupo, 't'], kind='stable').copy()
    df['dist'] = haversine(df['obj_lat'], df['obj_lng'], df['lat'], df['lng'])

    g = df.groupby(grupo, sort=False)['dist']
    df['delta_dist'] = g.diff().fillna(0.0)
    df['dist_med'] = g.transform(lambda s: s.rolling(JANELA, min_periods=1).median())
    df['dist_desv'] = g.transform(lambda s: s.rolling(JANELA, min_periods=2).std()).fillna(0.0)

    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    speed = df['speed'] if 'speed' in df else pd.Series(np.nan, index=df.index)
    df['speed'] = pd.to_numeric(speed, errors='coerce').fillna(0.0)
    disp = df.get('dispositivo', pd.Series('', index=df.index)).astype(str)
    df['iphone'] = disp.str.contains('iPhone', case=False).astype(int)
    return df


def montar_X(df: pd.DataFrame, accuracy_padrao: float | None = None) -> pd.DataFrame:
    X = df[FEATURES].astype(float)
    if accuracy_padrao is not None:
        X = X.fillna({'accuracy': accuracy_padrao})
    return X
