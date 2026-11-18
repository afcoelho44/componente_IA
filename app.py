"""
app.py — API Flask que corrige a distância do GPS até o objeto.

Dev:  flask --app app run --debug
Prod: gunicorn -b 0.0.0.0:5000 app:app

POST /corrigir  {"d_gps_m": 23.4, "accuracy": 6.0, "ambiente": "outdoor", "dispositivo": "Android"}
             -> {"dist_m": 19.8, "d_gps_m": 23.4, "erro_previsto": 3.6, "fora_da_faixa": false}
GET  /saude

Sem modelo treinado, /corrigir responde 503 e o Laravel usa a distância do GPS.
"""
import math
import os

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

from modelo_dist import FATOR_FAIXA, FEATURES, distancia_corrigida, montar_X

CAMINHO_MODELO = os.environ.get('MODELO', 'modelos/modelo.joblib')

app = Flask(__name__)


def carregar_modelo(caminho):
    if not os.path.exists(caminho):
        print(f'[aviso] modelo não encontrado em {caminho}: rode o treinar.py')
        return None
    pacote = joblib.load(caminho)
    if pacote.get('features') != FEATURES:
        raise RuntimeError(f'features do modelo {pacote.get("features")} != {FEATURES}: treine de novo')
    if pacote.get('sklearn') != sklearn.__version__:
        print(f'[aviso] modelo treinado com sklearn {pacote.get("sklearn")}, rodando {sklearn.__version__}')
    return pacote


PACOTE = carregar_modelo(CAMINHO_MODELO)


def numero(valor):
    """float finito ou None."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@app.get('/saude')
def saude():
    if PACOTE is None:
        return jsonify(ok=False, erro='modelo não treinado'), 503
    return jsonify(ok=True, treinado_em=PACOTE['treinado_em'], participantes=PACOTE['participantes'],
                   leituras=PACOTE['leituras'], faixa_d_gps=PACOTE['faixa_d_gps'],
                   sklearn=sklearn.__version__)


@app.post('/corrigir')
def corrigir():
    if PACOTE is None:
        return jsonify(erro='modelo não treinado'), 503

    dados = request.get_json(silent=True)
    if not isinstance(dados, dict):
        return jsonify(erro='envie um JSON com d_gps_m'), 400

    d_gps = numero(dados.get('d_gps_m'))
    if d_gps is None or d_gps < 0:
        return jsonify(erro='d_gps_m precisa ser um número >= 0'), 400

    linha = {
        'd_gps_m': d_gps,
        'accuracy': numero(dados.get('accuracy')),
        'ambiente': dados.get('ambiente') or '',
        'dispositivo': dados.get('dispositivo') or '',
    }

    # longe demais do que o modelo viu no treino: não corrige
    if d_gps > PACOTE['faixa_d_gps'][1] * FATOR_FAIXA:
        return jsonify(dist_m=round(d_gps, 2), d_gps_m=round(d_gps, 2), erro_previsto=0.0, fora_da_faixa=True)

    X = montar_X(pd.DataFrame([linha]), accuracy_padrao=PACOTE['accuracy_mediana'])
    erro = float(PACOTE['modelo'].predict(X)[0])
    dist = float(distancia_corrigida(d_gps, erro))

    return jsonify(dist_m=round(dist, 2), d_gps_m=round(d_gps, 2),
                   erro_previsto=round(erro, 2), fora_da_faixa=False)
