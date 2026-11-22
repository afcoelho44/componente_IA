"""
Testes da API de correção de distância.  Rodar: pytest -q
Treinam um modelo com o CSV falso (gerar_falso.py) numa pasta temporária.
"""
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

import app as api
from gerar_falso import gerar
from modelo_dist import FEATURES, montar_X

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def pacote(tmp_path_factory):
    pasta = tmp_path_factory.mktemp('ia')
    csv, modelo = pasta / 'falso.csv', pasta / 'modelo.joblib'
    gerar().to_csv(csv, index=False)
    subprocess.run([sys.executable, str(RAIZ / 'treinar.py'), str(csv), str(modelo)],
                   check=True, cwd=RAIZ, capture_output=True)
    return joblib.load(modelo)


@pytest.fixture
def cliente(pacote, monkeypatch):
    monkeypatch.setattr(api, 'PACOTE', pacote)
    api.app.config['TESTING'] = True
    return api.app.test_client()


def corrigir(cliente, **dados):
    return cliente.post('/corrigir', json=dados)


def test_saude(cliente):
    r = cliente.get('/saude')
    assert r.status_code == 200 and r.json['ok']


def test_sem_modelo_503(monkeypatch):
    monkeypatch.setattr(api, 'PACOTE', None)
    c = api.app.test_client()
    assert c.get('/saude').status_code == 503
    assert c.post('/corrigir', json={'d_gps_m': 10}).status_code == 503


def test_corrige_vies(cliente):
    # no CSV falso o GPS mede a mais (+2 m outdoor, +6 m indoor): na média,
    # a distância corrigida tem que ficar sem viés em relação à fita
    amostra = gerar(n_participantes=2, seed=1).query("fluxo == 'bruto'").sample(300, random_state=0)
    erros = []
    for _, l in amostra.iterrows():
        r = corrigir(cliente, d_gps_m=l.d_gps_m, accuracy=l.accuracy,
                     ambiente=l.ambiente, dispositivo=l.dispositivo_curto)
        assert r.status_code == 200 and not r.json['fora_da_faixa']
        erros.append(r.json['dist_m'] - l.distancia_m)
    vies_corrigido = sum(erros) / len(erros)
    vies_gps = (amostra.d_gps_m - amostra.distancia_m).mean()
    assert abs(vies_corrigido) < 1.0 < vies_gps


def test_iphone_sozinho(cliente):
    # uma linha só com iPhone: one-hot não pode perder colunas
    r = corrigir(cliente, d_gps_m=15, accuracy=12, ambiente='indoor', dispositivo='iPhone')
    assert r.status_code == 200


def test_accuracy_nula(cliente):
    assert corrigir(cliente, d_gps_m=15, accuracy=None, ambiente='outdoor').status_code == 200


@pytest.mark.parametrize('dados', [{}, {'d_gps_m': -1}, {'d_gps_m': 'abc'}, {'accuracy': 5}])
def test_entrada_invalida_400(cliente, dados):
    assert cliente.post('/corrigir', json=dados).status_code == 400


def test_sem_json_400(cliente):
    assert cliente.post('/corrigir', data='x').status_code == 400


def test_fora_da_faixa_nao_corrige(cliente, pacote):
    longe = pacote['faixa_d_gps'][1] * 3
    r = corrigir(cliente, d_gps_m=longe, accuracy=5, ambiente='outdoor', dispositivo='Android')
    assert r.json['fora_da_faixa'] and r.json['dist_m'] == pytest.approx(longe, abs=0.01)


def test_distancia_nunca_negativa(cliente):
    assert corrigir(cliente, d_gps_m=0, accuracy=15, ambiente='indoor').json['dist_m'] >= 0


def test_montar_X_mesmas_colunas():
    uma = montar_X(pd.DataFrame([{'d_gps_m': 5, 'accuracy': 3, 'ambiente': 'indoor', 'dispositivo': 'iPhone'}]))
    varias = montar_X(gerar(n_participantes=2).head(50))
    assert list(uma.columns) == list(varias.columns) == FEATURES
