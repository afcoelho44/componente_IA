"""
gerar_falso.py — CSV falso no formato do leituras_processadas.csv (analisar.py), para
testar o treino e a API sem os dados reais.

Uso: python gerar_falso.py dados/falso.csv

Cada participante fica parado ~60 s em cada marca da fita (0, 5, 10, 15, 20, 25 m),
em dois pontos. O GPS tem um desvio fixo por captura (como nos dados reais) mais ruído.
"""
import sys

import numpy as np
import pandas as pd

M_LAT = 111_320.0


def gerar(n_participantes=4, seed=0):
    rng = np.random.default_rng(seed)
    pontos = {'Ponto A': (-26.2535, -48.8555), 'Ponto B': (-26.2532, -48.8558)}
    linhas, cap = [], 0
    for p in range(1, n_participantes + 1):
        disp = 'iPhone' if p % 2 else 'Android'
        for local, (olat, olng) in pontos.items():
            m_lng = M_LAT * np.cos(np.radians(olat))
            for dist in (0, 5, 10, 15, 20, 25):
                cap += 1
                acc = rng.uniform(3, 6)
                desvio = rng.normal(0, 2.5, size=2)        # desvio fixo da captura (m)
                for k in range(60):
                    norte = -dist + desvio[0] + rng.normal(0, acc / 3)   # a fita vai para o sul
                    leste = desvio[1] + rng.normal(0, acc / 3)
                    linhas.append({
                        'captura_id': f'c{cap}', 'participante': f'P{p:02d}', 'local': local,
                        'ambiente': 'outdoor', 'distancia_m': float(dist), 'repeticao': 1,
                        'dispositivo_curto': disp, 't_ms': k * 1000, 't_gps_s': 1_700_000_000 + cap * 100 + k,
                        'lat': olat + norte / M_LAT, 'lng': olng + leste / m_lng,
                        'accuracy': round(acc + rng.normal(0, 0.3), 2),
                        'speed': abs(rng.normal(0, 0.2)) if disp == 'Android' else np.nan,
                        'obj_lat': olat, 'obj_lng': olng, 'avaliar': k >= 15,
                    })
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    saida = sys.argv[1] if len(sys.argv) > 1 else 'dados/falso.csv'
    gerar().to_csv(saida, index=False)
    print(f'CSV falso salvo em {saida}')
