import pandas as pd
import pytest

from app import app
from rotular import FEATURES, preparar

CSV = 'dados/samples.csv'


@pytest.fixture
def cliente():
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def janela():
    # TODO: ler CSV, pegar 4 linhas consecutivas de uma mesma sessão
    #       e devolver como lista de dicts: df.iloc[i:i+4].to_dict('records')
    #       cuidado: NaN do pandas -> troque por None antes (df.astype(object).where(df.notna(), None))
    ...


def test_saude(cliente):
    # TODO: GET /saude -> 200 e 'ok' True
    ...

def test_quatro_linhas_ok(cliente, janela):
    # TODO: POST /prever com {'linhas': janela} -> 200, 'dica' em quente/neutro/frio,
    #       soma das probabilidades ≈ 1
    ...

def test_tres_linhas_400(cliente, janela):
    # TODO: janela[:3] -> 400
    ...

def test_gps_parado_nulos(cliente, janela):
    # TODO: zerar speed e pôr gps_heading=None em todas -> ainda 200
    ...

def test_buraco_nao_confiavel(cliente, janela):
    # TODO: somar 2000 ao t_ms da última linha -> confiavel False
    ...

def test_mesmas_features_que_treino(janela):
    # TODO: preparar() no CSV inteiro vs preparar() só na janela:
    #       as FEATURES da última linha da janela devem ser iguais
    #       (pd.testing.assert_series_equal ou np.allclose com equal_nan=True)
    ...