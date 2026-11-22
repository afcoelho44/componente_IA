"""
Testes da API de classificação da proximidade.  Rodar: pytest -q
Treinam um modelo com o CSV falso (gerar_falso.py) numa pasta temporária.
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import app as api
from gerar_falso import gerar
from modelo_classe import CLASSES, FEATURES, classe_por_distancia, montar_X, preparar

RAIZ = Path(__file__).resolve().parents[1]
OBJ = {'lat': -26.2535, 'lng': -48.8555}
M_LAT = 111_320.0


@pytest.fixture(scope='session')
def pacote(tmp_path_factory):
    pasta = tmp_path_factory.mktemp('ia')
    csv, modelo = pasta / 'falso.csv', pasta / 'modelo.joblib'
    gerar().to_csv(csv, index=False)
    subprocess.run([sys.executable, str(RAIZ / 'treinar.py'), str(csv), str(modelo),
                    '--saida-validacao', str(pasta / 'validacao')],
                   check=True, cwd=RAIZ, capture_output=True)
    return joblib.load(modelo)


@pytest.fixture
def cliente(pacote, monkeypatch):
    monkeypatch.setattr(api, 'PACOTE', pacote)
    api.app.config['TESTING'] = True
    return api.app.test_client()


def leituras_a(metros_ao_sul, n=5, accuracy=4.0):
    lat = OBJ['lat'] - metros_ao_sul / M_LAT
    return [{'lat': lat, 'lng': OBJ['lng'], 'accuracy': accuracy, 'speed': 0, 't_ms': 1000 * i} for i in range(n)]


def classificar(cliente, leituras, dispositivo='Android'):
    return cliente.post('/classificar', json={'objeto': OBJ, 'leituras': leituras, 'dispositivo': dispositivo})


def test_limites():
    assert list(classe_por_distancia([0, 10, 10.01, 15, 15.01, 100])) == \
        ['quente', 'quente', 'morno', 'morno', 'frio', 'frio']
    assert classe_por_distancia(12) == 'morno'


def test_saude(cliente):
    r = cliente.get('/saude')
    assert r.status_code == 200 and r.json['ok'] and r.json['classes']


def test_sem_modelo_503(monkeypatch):
    monkeypatch.setattr(api, 'PACOTE', None)
    c = api.app.test_client()
    assert c.get('/saude').status_code == 503
    assert classificar(c, leituras_a(5)).status_code == 503


@pytest.mark.parametrize('metros, esperada', [(2, 'quente'), (22, 'frio')])
def test_classes_claras(cliente, metros, esperada):
    r = classificar(cliente, leituras_a(metros))
    assert r.status_code == 200
    assert r.json['classe'] == esperada and r.json['fonte'] == 'ia'
    assert set(r.json['probabilidades']) == set(CLASSES)
    assert sum(r.json['probabilidades'].values()) == pytest.approx(1, abs=0.01)


def test_dist_m(cliente):
    assert classificar(cliente, leituras_a(12)).json['dist_m'] == pytest.approx(12, abs=0.1)


def test_longe_demais_usa_regra(cliente, pacote):
    r = classificar(cliente, leituras_a(pacote['dist_max_treino'] * 3))
    assert r.json['fonte'] == 'regra' and r.json['classe'] == 'frio'


def test_uma_leitura_so_e_iphone(cliente):
    r = classificar(cliente, leituras_a(5, n=1), dispositivo='iPhone')
    assert r.status_code == 200


def test_campos_opcionais_nulos(cliente):
    leituras = [{'lat': l['lat'], 'lng': l['lng'], 'accuracy': None, 'speed': None} for l in leituras_a(8)]
    assert classificar(cliente, leituras).status_code == 200


def test_muitas_leituras_corta(cliente):
    assert classificar(cliente, leituras_a(5, n=200)).status_code == 200


@pytest.mark.parametrize('corpo', [
    {},
    {'objeto': OBJ},
    {'objeto': OBJ, 'leituras': []},
    {'objeto': {'lat': 'x', 'lng': 1}, 'leituras': leituras_a(5)},
    {'objeto': OBJ, 'leituras': [{'lat': 200, 'lng': 0}]},
    {'objeto': OBJ, 'leituras': ['abc']},
])
def test_entrada_invalida_400(cliente, corpo):
    assert cliente.post('/classificar', json=corpo).status_code == 400


def test_sem_json_400(cliente):
    assert cliente.post('/classificar', data='x').status_code == 400


def test_features_so_usam_o_passado():
    # mudar uma leitura futura não pode mudar as features das anteriores
    df = gerar(n_participantes=1).head(30).rename(columns={'t_gps_s': 't'})
    antes = preparar(df)[FEATURES].iloc[:20].to_numpy()
    df.loc[25, 'lat'] += 0.001
    depois = preparar(df)[FEATURES].iloc[:20].to_numpy()
    assert np.allclose(antes, depois)


def test_montar_X_colunas():
    df = preparar(pd.DataFrame([{'captura_id': 1, 't': 0, 'lat': -26.25, 'lng': -48.85,
                                 'obj_lat': -26.25, 'obj_lng': -48.85, 'accuracy': 3}]))
    assert list(montar_X(df).columns) == FEATURES
