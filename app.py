"""
app.py — API Flask do quente/frio.
Dev:  flask --app app run --debug
Prod: gunicorn -b 0.0.0.0:5000 app:app
"""
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

from rotular import ACCURACY_MAX, FEATURES, GPS_IDADE_MAX, preparar

CAMINHO_MODELO = os.environ.get('MODELO', 'modelos/modelo.joblib')
MIN_LINHAS = 4   # delta_dist_3s = diff(3) precisa de 4 linhas
COLUNAS_OBRIGATORIAS = [
    # TODO: listar as colunas que o preparar() lê (sessao, t_ms, gps_t_ms, lat, lng,
    #       alvo_lat, alvo_lng, accuracy, speed, gps_heading, bussola_media, ...)
]

app = Flask(__name__)


def carregar_modelo(caminho):
    # TODO 1: pacote = joblib.load(caminho)
    # TODO 2: se pacote['features'] != FEATURES -> raise RuntimeError
    # TODO 3: se pacote['sklearn'] != sklearn.__version__ -> print de aviso
    # TODO 4: return pacote
    ...


PACOTE = carregar_modelo(CAMINHO_MODELO)   # carrega 1 vez, na subida


def validar_entrada(dados):
    """Retorna (lista_de_linhas, None) ou (None, mensagem_de_erro)."""
    # TODO 1: dados precisa ser dict com chave 'linhas' do tipo list
    # TODO 2: len(linhas) >= MIN_LINHAS
    # TODO 3: toda linha tem todas as COLUNAS_OBRIGATORIAS (diga quais faltam)
    # TODO 4: todas as linhas têm a mesma 'sessao'
    ...


def montar_df(linhas):
    # TODO 1: df = pd.DataFrame(linhas)
    # TODO 2: converter todas as colunas menos 'sessao' com
    #         pd.to_numeric(errors='coerce')  <- sem isso, null vira texto
    # TODO 3: return df
    ...


def eh_confiavel(ultima):
    # TODO: False se accuracy > ACCURACY_MAX, gps_idade_s > GPS_IDADE_MAX
    #       ou delta_dist_3s for NaN (buraco no tempo / segmento curto)
    ...


@app.get('/saude')
def saude():
    # TODO: jsonify com ok, treinado_em, sessoes, sklearn
    ...


@app.post('/prever')
def prever():
    # TODO 1: linhas, erro = validar_entrada(request.get_json(silent=True))
    #         se erro -> return jsonify({'erro': erro}), 400
    # TODO 2: prep = preparar(montar_df(linhas))
    # TODO 3: X = prep[FEATURES].iloc[[-1]]      (colchetes duplos!)
    # TODO 4: proba = PACOTE['modelo'].predict_proba(X)[0]
    #         probabilidades = dict(zip(modelo.classes_, proba)) — arredonde e
    #         converta para float (numpy float não serializa bem em JSON)
    # TODO 5: dica = classe com maior probabilidade
    # TODO 6: ultima = prep.iloc[-1]
    #         return jsonify(dica=..., probabilidades=..., 
    #                        confiavel=eh_confiavel(ultima), dist=round(...))
    ...