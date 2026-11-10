"""
treinar.py — treina o Random Forest quente/frio.
Uso: python treinar.py dados/dataset.csv modelos/modelo.joblib
"""
import argparse
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                            confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from rotular import FEATURES

CLASSES = ['quente', 'neutro', 'frio']   # ordem fixa para a matriz de confusão


def novo_modelo():
    # TODO: devolver um RandomForestClassifier com os parâmetros da Etapa 1
    #       (n_estimators, class_weight, min_samples_leaf, random_state)
    ...


def validar(X, y, grupos):
    # TODO 1: se houver menos de 2 sessões, avisar e retornar sem validar
    # TODO 2: y_prev = cross_val_predict(novo_modelo(), X, y,
    #                                    groups=grupos, cv=LeaveOneGroupOut())
    #         (cada sessão vira teste uma vez, treinando nas outras)
    # TODO 3: imprimir classification_report(y, y_prev)
    # TODO 4: imprimir a confusion_matrix com labels=CLASSES
    #         (dica: pd.DataFrame(..., index=CLASSES, columns=CLASSES) fica legível)
    # TODO 5: imprimir balanced_accuracy_score
    # TODO 6 (opcional): linha de base — prever 'quente' se delta_dist_3s < -1.5,
    #         'frio' se > 1.5, senão 'neutro', e imprimir o mesmo relatório
    ...


def main():
    # TODO 1: argparse com entrada (dataset.csv) e saida (modelo.joblib)
    # TODO 2: ler o csv; X = df[FEATURES], y = df['rotulo'], grupos = df['sessao']
    # TODO 3: imprimir quantas linhas por sessão e por rótulo (pd.crosstab)
    # TODO 4: validar(X, y, grupos)
    # TODO 5: treinar o modelo final com novo_modelo().fit(X, y) em TUDO
    # TODO 6: joblib.dump de um dicionário com:
    #         'modelo', 'features', 'sklearn' (versão), 'rotulagem'
    #         (horizonte/margem/suavizar), 'treinado_em', 'sessoes' (quantidade)
    # TODO 7 (opcional): imprimir rf.feature_importances_ ordenado — bom pro relatório
    ...


if __name__ == '__main__':
    main()