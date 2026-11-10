"""
rotular.py — prepara o dataset quente/frio a partir do samples.csv do Laravel.

Uso:
    pip install pandas numpy
    python rotular.py samples.csv dataset.csv
    python rotular.py samples.csv dataset.csv --horizonte 5 --margem 2 --suavizar 5

O que faz:
  1. limpa os dados (heading/speed falsos do GPS parado, fix velho, accuracy ruim)
  2. calcula features que usam só o PRESENTE e o PASSADO (as mesmas que a API
     vai calcular ao vivo) -> função preparar()
  3. cria o rótulo olhando o FUTURO suavizado da distância -> função rotular()
     quente  : a distância vai diminuir mais que `margem` metros em `horizonte` s
     frio    : a distância vai aumentar mais que `margem`
     neutro  : no meio disso
"""

import argparse

import numpy as np
import pandas as pd

RAIO_TERRA = 6_371_000.0

# --- limites de limpeza -----------------------------------------------------
SPEED_MIN = 0.5          # m/s — abaixo disso speed/heading do GPS são ruído
ACCURACY_MAX = 20.0      # m   — fix pior que isso não entra no rótulo
GPS_IDADE_MAX = 3.0      # s   — fix mais velho que isso não entra no rótulo
GAP_MAX_MS = 1500        # buraco maior que isso quebra a sessão em segmentos

# Colunas que entram no Random Forest (o que a API vai receber/calcular ao vivo).
# Não entram: ids, lat/lng absolutos, alvo, e nada que olhe o futuro.
FEATURES = [
    'dist',                 # distância bruta até o alvo (m)
    'accuracy',
    'gps_idade_s',
    'speed_limpa',
    'diff_rumo_bussola',    # alvo em relação para onde o celular aponta (-180..180)
    'diff_rumo_gps',        # alvo em relação para onde a pessoa anda (-180..180)
    'bussola_desvio',
    'beta', 'gamma',
    'giro_alpha_soma', 'giro_beta_media', 'giro_gamma_media',
    'acc_mag_media', 'acc_mag_desvio',
    'delta_dist_3s',        # quanto a distância mudou nos últimos 3 s (passado)
    'acc_desvio_media_3s',  # está andando há alguns segundos?
]


# ---------------------------------------------------------------------------
# ângulos e geografia
# ---------------------------------------------------------------------------
def normalizar180(graus):
    """qualquer ângulo -> -180..180 (positivo = alvo à direita)"""
    return (graus + 180.0) % 360.0 - 180.0


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * RAIO_TERRA * np.arcsin(np.sqrt(h))


def rumo(lat1, lng1, lat2, lng2):
    """rumo de (1) para (2), 0..360, 0 = norte, sentido horário"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng2 - lng1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# 1 + 2: limpeza e features (presente/passado) — reaproveitar na API Flask
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['sessao', 't_ms']).reset_index(drop=True).copy()

    # segmentos contínuos: um buraco no tempo não pode virar "deslocamento"
    novo = (df['sessao'] != df['sessao'].shift()) | (df['t_ms'].diff() > GAP_MAX_MS)
    df['segmento'] = novo.cumsum()

    # idade do fix do GPS no momento da linha
    df['gps_idade_s'] = (df['t_ms'] - df['gps_t_ms']) / 1000.0

    # speed/heading do GPS só valem andando de verdade
    andando_gps = df['speed'].fillna(0) >= SPEED_MIN
    df['speed_limpa'] = df['speed'].where(andando_gps, 0.0)
    df['gps_heading_limpo'] = df['gps_heading'].where(andando_gps)  # NaN se parado

    # geometria até o alvo
    df['dist'] = haversine(df['lat'], df['lng'], df['alvo_lat'], df['alvo_lng'])
    df['rumo_alvo'] = rumo(df['lat'], df['lng'], df['alvo_lat'], df['alvo_lng'])
    df['diff_rumo_bussola'] = normalizar180(df['rumo_alvo'] - df['bussola_media'])
    df['diff_rumo_gps'] = normalizar180(df['rumo_alvo'] - df['gps_heading_limpo'])

    # features de passado curto (por segmento)
    g = df.groupby('segmento', group_keys=False)
    df['delta_dist_3s'] = g['dist'].diff(3)
    df['acc_desvio_media_3s'] = g['acc_mag_desvio'].transform(
        lambda s: s.rolling(3, min_periods=1).mean())

    return df


# ---------------------------------------------------------------------------
# 3: rótulo (usa o futuro — só existe no treino)
# ---------------------------------------------------------------------------
def rotular(df: pd.DataFrame, horizonte: int = 5, margem: float = 2.0,
            suavizar: int = 5) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby('segmento', group_keys=False)

    # distância suavizada (janela centrada) — tira o "pula-pula" do GPS
    df['dist_suav'] = g['dist'].transform(
        lambda s: s.rolling(suavizar, center=True, min_periods=1).median())

    # quanto a distância suavizada vai mudar nos próximos `horizonte` segundos
    df['delta_futuro'] = (
        df.groupby('segmento')['dist_suav'].shift(-horizonte) - df['dist_suav'])

    df['rotulo'] = np.select(
        [df['delta_futuro'] < -margem, df['delta_futuro'] > margem],
        ['quente', 'frio'],
        default='neutro',
    )

    # sem futuro (fim do segmento) ou GPS ruim -> sem rótulo
    invalida = (
        df['delta_futuro'].isna()
        | (df['accuracy'] > ACCURACY_MAX)
        | (df['gps_idade_s'] > GPS_IDADE_MAX)
    )
    df.loc[invalida, 'rotulo'] = None
    return df


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Rotula o samples.csv em quente/frio/neutro')
    p.add_argument('entrada')
    p.add_argument('saida')
    p.add_argument('--horizonte', type=int, default=5, help='segundos à frente (padrão 5)')
    p.add_argument('--margem', type=float, default=2.0, help='metros (padrão 2.0)')
    p.add_argument('--suavizar', type=int, default=5, help='janela da mediana (padrão 5)')
    a = p.parse_args()

    bruto = pd.read_csv(a.entrada)
    df = rotular(preparar(bruto), a.horizonte, a.margem, a.suavizar)

    rotulado = df[df['rotulo'].notna()]
    colunas = ['id', 'sessao', 'segmento', 't_ms', *FEATURES,
               'dist_suav', 'delta_futuro', 'rotulo']
    rotulado[colunas].to_csv(a.saida, index=False)

    print(f'{len(bruto)} linhas lidas, {len(rotulado)} rotuladas '
          f'({len(bruto) - len(rotulado)} sem rótulo: fim de segmento ou GPS ruim)')
    print('\nRótulos por sessão:')
    print(pd.crosstab(rotulado['sessao'].str[:8], rotulado['rotulo'], margins=True,
                      margins_name='total'))
    print('\nDistância (m) por sessão:')
    print(df.groupby(df['sessao'].str[:8])['dist'].agg(['min', 'max']).round(1))
    print(f'\nSalvo em {a.saida}')


if __name__ == '__main__':
    main()
