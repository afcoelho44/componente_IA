"""
treinar.py — treina o Random Forest que classifica a proximidade (quente/morno/frio)
e compara com a regra de distância.

Uso:
    python treinar.py dados/leituras_processadas.csv modelos/modelo.joblib
    python treinar.py dados/leituras_processadas.csv modelos/modelo.joblib \
        --excluir-local "Entre o bloco F e o RU" --saida-validacao dados/validacao

Entrada: leituras_processadas.csv do experimento/analisar.py (uma linha por leitura,
com obj_lat/obj_lng = média das leituras a 0 m e distancia_m = fita).

Validação: separa por participante quando há 2 ou mais; senão, por ponto (local).
Compara, nos mesmos dados de teste:
  - Regra          : limites aplicados direto na distância do GPS
  - Random Forest  : modelo treinado sem aquele participante/ponto
Métricas: acurácia, F1 macro, precisão/recall por classe, matriz de confusão e
trocas de classe por captura (estabilidade da dica com o celular parado).
"""
import argparse
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from modelo_classe import CLASSES, FEATURES, LIMITES, classe_por_distancia, montar_X, preparar


def novo_modelo():
    # min_samples_leaf alto: leituras da mesma captura são quase iguais e o modelo decoraria
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight='balanced',
                                  random_state=42, n_jobs=-1)


def carregar(caminho, ambiente=None, excluir_locais=()):
    df = pd.read_csv(caminho)
    obrig = {'captura_id', 'lat', 'lng', 'accuracy', 'obj_lat', 'obj_lng', 'distancia_m'}
    falta = obrig - set(df.columns)
    if falta:
        raise SystemExit(f'faltam colunas no CSV: {sorted(falta)} (use o leituras_processadas.csv do analisar.py)')

    df['participante'] = df.get('participante', pd.Series('sem_id', index=df.index)).fillna('sem_id').astype(str)
    df['local'] = df.get('local', pd.Series('sem_local', index=df.index)).astype(str)
    df['dispositivo'] = df.get('dispositivo_curto', df.get('dispositivo', pd.Series('', index=df.index)))
    df['t'] = df['t_gps_s'] if 't_gps_s' in df else df['t_ms'] / 1000.0

    if ambiente:
        df = df[df['ambiente'].astype(str).str.lower() == ambiente.lower()]
    if excluir_locais:
        df = df[~df['local'].isin(excluir_locais)]
    df = df.dropna(subset=['lat', 'lng', 'obj_lat', 'obj_lng', 'distancia_m'])

    # features com o histórico desde o início da captura; avaliação só depois do aquecimento
    df = preparar(df)
    if 'avaliar' in df:
        df = df[df['avaliar'].astype(str).str.lower() == 'true']
    df['classe_real'] = classe_por_distancia(df['distancia_m'])
    return df.reset_index(drop=True)


def trocas_por_captura(df, coluna):
    return float(df.groupby('captura_id')[coluna].apply(lambda s: (s != s.shift()).sum() - 1).mean())


def resumo(df, coluna):
    return {
        'acuracia': accuracy_score(df['classe_real'], df[coluna]),
        'f1_macro': f1_score(df['classe_real'], df[coluna], labels=CLASSES, average='macro', zero_division=0),
        'trocas_por_captura': trocas_por_captura(df, coluna),
    }


def validar(df, acc_med):
    if df['participante'].nunique() >= 2:
        coluna_grupo = 'participante'
    elif df['local'].nunique() >= 2:
        coluna_grupo = 'local'
    else:
        print('\nSó 1 participante e 1 ponto: sem validação (precisa de 2 grupos).')
        return None, None

    grupos = df[coluna_grupo]
    n = grupos.nunique()
    X = montar_X(df, acc_med)
    df = df.copy()
    df['regra'] = classe_por_distancia(df['dist'])
    df['rf'] = None
    for treino, teste in GroupKFold(n_splits=min(5, n)).split(X, df['classe_real'], grupos):
        rf = novo_modelo().fit(X.iloc[treino], df['classe_real'].iloc[treino])
        df.loc[df.index[teste], 'rf'] = rf.predict(X.iloc[teste])

    print(f'\n== Validação deixando um(a) {coluna_grupo} de fora ({n} grupos) ==')
    tabela = pd.DataFrame([{'metodo': 'Regra', **resumo(df, 'regra')},
                           {'metodo': 'Random Forest', **resumo(df, 'rf')}])
    print(tabela.round(2).to_string(index=False))
    for nome, col in [('Regra', 'regra'), ('Random Forest', 'rf')]:
        cm = pd.DataFrame(confusion_matrix(df['classe_real'], df[col], labels=CLASSES),
                          index=[f'real {c}' for c in CLASSES], columns=CLASSES)
        print(f'\n-- {nome}: matriz de confusão (linha = real, coluna = previsto)')
        print(cm.to_string())
        print(classification_report(df['classe_real'], df[col], labels=CLASSES, zero_division=0, digits=2))

    por_grupo = (df.groupby(coluna_grupo)
                   .apply(lambda g: pd.Series({f'{m}_{k}': v for m in ('regra', 'rf') for k, v in resumo(g, m).items()}))
                   .reset_index())
    return tabela, (df, por_grupo)


def salvar_validacao(pasta, tabela, extras):
    os.makedirs(pasta, exist_ok=True)
    df, por_grupo = extras
    tabela.to_csv(os.path.join(pasta, 'resumo.csv'), index=False)
    por_grupo.to_csv(os.path.join(pasta, 'por_grupo.csv'), index=False)
    for col in ('regra', 'rf'):
        pd.DataFrame(confusion_matrix(df['classe_real'], df[col], labels=CLASSES),
                     index=CLASSES, columns=CLASSES).to_csv(os.path.join(pasta, f'matriz_{col}.csv'))
    df[['captura_id', 'participante', 'local', 'dispositivo', 'distancia_m', 't', *FEATURES,
        'classe_real', 'regra', 'rf']].to_csv(os.path.join(pasta, 'previsoes.csv'), index=False)
    print(f'\nValidação salva em {pasta}/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('--ambiente', default='outdoor', help='filtra o ambiente (vazio = todos)')
    ap.add_argument('--excluir-local', action='append', default=[], help='ponto a deixar de fora (repetível)')
    ap.add_argument('--saida-validacao', default=None, help='pasta para salvar as tabelas da validação')
    a = ap.parse_args()

    df = carregar(a.entrada, a.ambiente or None, a.excluir_local)
    if df.empty:
        raise SystemExit('nenhuma leitura sobrou depois dos filtros')

    print(f'{len(df)} leituras, {df["participante"].nunique()} participante(s), '
          f'{df["local"].nunique()} ponto(s), {df["captura_id"].nunique()} capturas')
    print('\nLeituras por distância da fita e classe:')
    print(pd.crosstab(df['distancia_m'], df['classe_real']).reindex(columns=CLASSES, fill_value=0).to_string())
    sem = [c for c in CLASSES if c not in set(df['classe_real'])]
    if sem:
        print(f'\nAtenção: sem amostras das classes {sem} — o modelo nunca vai prevê-las.')

    acc_med = float(df['accuracy'].median()) if df['accuracy'].notna().any() else 5.0
    tabela, extras = validar(df, acc_med)
    if tabela is not None and a.saida_validacao:
        salvar_validacao(a.saida_validacao, tabela, extras)

    modelo = novo_modelo().fit(montar_X(df, acc_med), df['classe_real'])
    imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\n== Importância das features ==')
    print(imp.round(3).to_string())

    os.makedirs(os.path.dirname(a.saida) or '.', exist_ok=True)
    joblib.dump({
        'modelo': modelo,
        'features': FEATURES,
        'classes': [str(c) for c in modelo.classes_],
        'limites': LIMITES,
        'sklearn': sklearn.__version__,
        'treinado_em': datetime.now().isoformat(timespec='seconds'),
        'participantes': int(df['participante'].nunique()),
        'pontos': sorted(df['local'].unique().tolist()),
        'leituras': int(len(df)),
        'accuracy_mediana': acc_med,
        'dist_max_treino': float(df['dist'].max()),
        'validacao': None if tabela is None else tabela.round(4).to_dict(orient='records'),
    }, a.saida)
    print(f'\nModelo salvo em {a.saida}')


if __name__ == '__main__':
    main()
