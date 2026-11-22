"""
gerar_falso.py — CSV falso no formato do erro_distancia_por_leitura.csv, para testar
o treino e a API antes da coleta real com fita.

Uso: python gerar_falso.py dados/falso.csv

Regra embutida (o modelo tem que "descobrir"):
  outdoor: o GPS mede +2 m a mais;  indoor: +6 m;  iPhone: -1 m;  ruído ~ accuracy / 2
"""
import sys

import numpy as np
import pandas as pd


def gerar(n_participantes=6, seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    cap = 0
    for p in range(1, n_participantes + 1):
        disp = 'iPhone' if p % 2 else 'Android'
        for amb in ('indoor', 'outdoor'):
            for dist in (5, 10, 15, 20, 25, 30):
                for rep in (1, 2, 3):
                    cap += 1
                    acc = rng.uniform(3, 8) if amb == 'outdoor' else rng.uniform(8, 18)
                    vies = (2 if amb == 'outdoor' else 6) - (1 if disp == 'iPhone' else 0)
                    for t in range(30):
                        d_gps = max(0.0, dist + vies + rng.normal(0, acc / 2))
                        for fluxo in ('bruto', 'kalman'):
                            linhas.append({
                                'fluxo': fluxo, 'captura_id': cap, 'participante': f'P{p:02d}',
                                'dispositivo_curto': disp, 'ambiente': amb, 'local': 'Ponto A',
                                'distancia_m': dist, 'repeticao': rep, 't_ms': t * 1000,
                                'accuracy': round(acc, 1), 'd_gps_m': d_gps, 'erro_m': d_gps - dist,
                            })
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    saida = sys.argv[1] if len(sys.argv) > 1 else 'dados/falso.csv'
    gerar().to_csv(saida, index=False)
    print(f'CSV falso salvo em {saida}')
