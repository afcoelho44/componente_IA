"""
app.py — API Flask que classifica a proximidade do aluno ao objeto (quente/morno/frio).

Dev:  flask --app app run --debug
Prod: gunicorn -b 0.0.0.0:5000 app:app

POST /classificar
    {"objeto": {"lat": -26.25, "lng": -48.85},
     "leituras": [{"lat": ..., "lng": ..., "accuracy": 4.2, "speed": 0.8, "t_ms": 1732300000000}, ...],
     "dispositivo": "Android"}
    -> {"classe": "morno", "probabilidades": {"quente": 0.2, "morno": 0.7, "frio": 0.1},
        "dist_m": 12.3, "fonte": "ia"}

    "leituras" = as últimas leituras do GPS (mais antiga primeiro; até 20). A classe é da última.
    Longe demais do que o modelo viu no treino, responde com a regra de distância (fonte = "regra").
GET  /saude

Sem modelo treinado, /classificar responde 503 e o Laravel usa a regra.
"""
import math
import os

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

from modelo_classe import CLASSES, FEATURES, LIMITES, classe_por_distancia, montar_X, preparar

CAMINHO_MODELO = os.environ.get('MODELO', 'modelos/modelo.joblib')
MAX_LEITURAS = 20
FATOR_FAIXA = 1.5   # acima de FATOR_FAIXA × maior distância vista no treino, usa a regra

app = Flask(__name__)


def carregar_modelo(caminho):
    if not os.path.exists(caminho):
        print(f'[aviso] modelo não encontrado em {caminho}: rode o treinar.py')
        return None
    pacote = joblib.load(caminho)
    if pacote.get('features') != FEATURES or [tuple(l) for l in pacote.get('limites', [])] != [tuple(l) for l in LIMITES]:
        print(f'[aviso] modelo em {caminho} é de outra versão (features/limites diferentes): treine de novo')
        return None
    if pacote.get('sklearn') != sklearn.__version__:
        print(f'[aviso] modelo treinado com sklearn {pacote.get("sklearn")}, rodando {sklearn.__version__}')
    return pacote


PACOTE = carregar_modelo(CAMINHO_MODELO)


def numero(valor, minimo=None, maximo=None):
    """float finito dentro da faixa, ou None."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or (minimo is not None and v < minimo) or (maximo is not None and v > maximo):
        return None
    return v


def ler_entrada(dados):
    """Valida o JSON. Devolve (DataFrame das leituras, None) ou (None, mensagem de erro)."""
    if not isinstance(dados, dict):
        return None, 'envie um JSON com objeto e leituras'

    obj = dados.get('objeto')
    obj_lat = numero(obj.get('lat'), -90, 90) if isinstance(obj, dict) else None
    obj_lng = numero(obj.get('lng'), -180, 180) if isinstance(obj, dict) else None
    if obj_lat is None or obj_lng is None:
        return None, 'objeto precisa de lat e lng válidos'

    leituras = dados.get('leituras')
    if not isinstance(leituras, list) or not leituras:
        return None, 'leituras precisa ser uma lista com pelo menos 1 leitura'
    leituras = leituras[-MAX_LEITURAS:]

    linhas = []
    for i, l in enumerate(leituras):
        if not isinstance(l, dict):
            return None, f'leitura {i} inválida'
        lat, lng = numero(l.get('lat'), -90, 90), numero(l.get('lng'), -180, 180)
        if lat is None or lng is None:
            return None, f'leitura {i} sem lat/lng válidos'
        t_ms = numero(l.get('t_ms'))
        linhas.append({
            'grupo': 0,
            't': (t_ms / 1000.0) if t_ms is not None else float(i),
            'lat': lat, 'lng': lng,
            'accuracy': numero(l.get('accuracy'), 0),
            'speed': numero(l.get('speed'), 0),
            'obj_lat': obj_lat, 'obj_lng': obj_lng,
            'dispositivo': str(dados.get('dispositivo') or ''),
        })
    return pd.DataFrame(linhas), None


@app.get('/saude')
def saude():
    if PACOTE is None:
        return jsonify(ok=False, erro='modelo não treinado'), 503
    return jsonify(ok=True, treinado_em=PACOTE['treinado_em'], participantes=PACOTE['participantes'],
                   pontos=PACOTE['pontos'], leituras=PACOTE['leituras'], classes=PACOTE['classes'],
                   limites=PACOTE['limites'], dist_max_treino=PACOTE['dist_max_treino'],
                   validacao=PACOTE.get('validacao'), sklearn=sklearn.__version__)


@app.post('/classificar')
def classificar():
    if PACOTE is None:
        return jsonify(erro='modelo não treinado'), 503

    df, erro = ler_entrada(request.get_json(silent=True))
    if erro:
        return jsonify(erro=erro), 400

    df = preparar(df, grupo='grupo')
    atual = df.iloc[[-1]]
    dist = float(atual['dist'].iloc[0])

    # longe demais do que o modelo viu: a árvore não extrapola, então usa a regra
    if dist > PACOTE['dist_max_treino'] * FATOR_FAIXA:
        classe = str(classe_por_distancia(dist))
        return jsonify(classe=classe, probabilidades={c: float(c == classe) for c in CLASSES},
                       dist_m=round(dist, 2), fonte='regra')

    X = montar_X(atual, accuracy_padrao=PACOTE['accuracy_mediana'])
    modelo = PACOTE['modelo']
    probs = dict(zip(PACOTE['classes'], modelo.predict_proba(X)[0]))
    classe = max(probs, key=probs.get)
    return jsonify(classe=classe, probabilidades={c: round(float(probs.get(c, 0.0)), 3) for c in CLASSES},
                   dist_m=round(dist, 2), fonte='ia')
