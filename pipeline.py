"""
pipeline.py
===========
Pipeline completo de Treinamento, Otimização de Hiperparâmetros (Grid Search)
e Validação Cruzada K-Fold Estratificada para o modelo LSTM de Tradução de Libras.

Métricas de Avaliação:
- Acurácia (Accuracy)
- F1-Score Macro (F1-Macro)
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from feature_extraction import import_from_csv
from landmark_augmentation import gerar_amostras_aumentadas
from utils.constants import (
    DATASET_TESTE_CSV,
    DATASET_TREINO_CSV,
    ENCODER_PATH,
    KMEANS_PATH,
    LSTM_PATH,
    LSTM_PATH_AUG,
    MATRIZ_KMEANS_PATH,
    MATRIZ_LSTM_PATH,
    OUTPUTS_DIR,
    PARAM_GRID_KMEANS,
    PARAM_GRID_LSTM,
    PREDICOES_KMEANS_PATH,
    PREDICOES_LSTM_PATH,
    SEED,
)

tf.keras.backend.clear_session()

def splitTrainValidation(x_fit, y_fit, groups_fit):
    """
    Separa dados de treino dos dados de teste para aplicar
    o EarlyStopping no GridSearch corretamente
    """
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    X_dummy = np.zeros(len(y_fit))
    train_idx, val_idx = next(
        cv.split(X_dummy, y_fit, groups=groups_fit)
    )

    x_train = x_fit[train_idx]
    y_train = y_fit[train_idx]
    groups_train = groups_fit[train_idx]

    x_val = x_fit[val_idx]    
    y_val = y_fit[val_idx]    
    groups_val = groups_fit[val_idx]

    overlap = set(groups_train).intersection(set(groups_val))
    if overlap:
        raise ValueError(f'Overlap entre grupos de Treino e Validação: {overlap}')    

    return x_train, y_train, groups_train, x_val, y_val, groups_val


def create_lstm_model(
    input_shape, num_classes, lstm_units_1=128, 
    lstm_units_2=64, dense_units=64, dropout_rate=0.3, learning_rate=0.001
):
    """
        Constrói e compila o modelo LSTM com os hiperparâmetros fornecidos.
    """

    model = Sequential([
        # Camada de entrada do Modelo
        Input(shape=input_shape),
        # Primeira camada LSTM, aprende padrões temporais
        # ao longo dos frames
        LSTM(units=lstm_units_1, return_sequences=True),
        # Reduz overfitting, desliga parte
        # das ativações aleatoriamente
        Dropout(dropout_rate),
        # Segunda camada LSTM, recebe padrões temporais
        # da primeira LSTM, gera representação final da
        # sequência completa
        LSTM(units=lstm_units_2, return_sequences=False),
        # Reduz dependência excessiva de características
        # específicas do treino
        Dropout(dropout_rate),
        # Combina características temporais, ReLU
        # permite a representação de relações não linerares
        # entre características extraídas da sequência
        Dense(units=dense_units, activation="relu"),
        # Auxiliar na generalização do modelo
        Dropout(dropout_rate),
        # Camada de Saída
        # Um neurônio por classe
        # Softmax permite a classificação multiclasse
        Dense(units=num_classes, activation="softmax")
    ])
    # Adequado para o treinamento de redes recorrentes
    optimizer = Adam(learning_rate=learning_rate)

    # sparse_categorical_crossentropy: classes como inteiros (0, 1, 2, ...)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


class SequenciaParaHistograma(BaseEstimator, TransformerMixin):
    """
    Agrupa os frames em poses (K-Means) e resume cada sequência num histograma de
    ocupação + transição entre poses consecutivas.
    """

    def __init__(self, n_clusters=110):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        frames_treino = X.reshape(-1, X.shape[-1])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=SEED,
                              n_init=10)
        self.kmeans_.fit(frames_treino)
        return self

    def transform(self, X):
        n_clusters = self.n_clusters
        vetores = np.zeros(
            (len(X), n_clusters + n_clusters * n_clusters))

        for i, seq in enumerate(X):
            clusters_seq = self.kmeans_.predict(seq)

            for c in clusters_seq:
                vetores[i, c] += 1
            vetores[i, :n_clusters] /= len(clusters_seq)

            for c_atual, c_seguinte in zip(clusters_seq[:-1],
                                           clusters_seq[1:]):
                vetores[i, n_clusters + c_atual * n_clusters +
                        c_seguinte] += 1
            n_transicoes = max(len(clusters_seq) - 1, 1)
            vetores[i, n_clusters:] /= n_transicoes

        return vetores


def criar_pipeline_kmeans():
    return Pipeline([
        ("compactar", SequenciaParaHistograma()),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(max_depth=None,
                                      class_weight="balanced",
                                      random_state=SEED)),
    ])


def applyGridSearch(model_used, params, scoring_method, X_fit, y_fit, groups, X_val, y_val,
                    usa_validation_data=True):
    """
    Executa GridSearchCV para um dado modelo (pipeline) e espaço de parâmetros.

    Parâmetros:
        model_used              : Estimador base.
        params                  : dicionário de hiperparâmetros.
        scoring_method          : métrica de avaliação.
        X_fit, y_fit, groups    : dados de treino.
        X_val, y_val            : dados de validação
        usa_validation_data     : False pra modelos sklearn puros (não usam EarlyStopping)

    Retorna:
        best_estimator_         : melhor modelo encontrado.
        best_params_            : melhor combinação de hiperparâmetros.
        best_score_             : melhor score obtido na validação cruzada.
    """

    if usa_validation_data:
        model_used.set_params(
            fit__validation_data=(X_val, y_val)
        )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # TODO Adicionar dados de validação
    grid = GridSearchCV(
        estimator=model_used,
        param_grid=params,
        cv=cv,
        scoring=scoring_method,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_fit, y_fit, groups=groups)
    return grid.best_estimator_, grid.best_params_, grid.best_score_



def avaliar_modelo_teste(model_name, model, best_params, label_encoder, X_test, y_test,
                         usa_predict_proba=True,
                         predicoes_path=PREDICOES_LSTM_PATH,
                         matriz_path=MATRIZ_LSTM_PATH):
    """
    Realiza o Teste do Modelo, coleta as Métricas (F1-Score Macro e Acurácia Geral)
    e gera a Matriz de Confusão da predição do modelo
    """
    # Realiza as predições do modelo
    if usa_predict_proba:
        y_pred_probs = model.predict_proba(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = model.predict(X_test)

    # Valor Previsto -> Gesto do LabelEncoder
    y_pred_str = label_encoder.inverse_transform(y_pred)

    # Gesto -> Valor do LabelEncoder
    y_test_encoded = label_encoder.transform(y_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    f1_mac = f1_score(y_test_encoded, y_pred, average="macro", zero_division=0)
    report_text = str(classification_report(y_test, y_pred_str, zero_division=0, output_dict=False))

    # Salva resultados em outputs/
    file_ = open(f'{OUTPUTS_DIR}/results_{model_name}.txt', 'w')
    file_.truncate(0)
    file_.write(f"Total de Amostras de Teste: {len(y_test)}\n")
    file_.write(f"Acurácia no Teste: {acc * 100:.2f}%\n")
    file_.write(f"F1-Score Macro:             {f1_mac * 100:.2f}%\n")

    file_.write(f"Hiperparametros do {model_name}: {best_params.items()}")

    file_.write("\nRelatório de Classificação por Classe:\n")
    file_.write(report_text)
    file_.close()

    # Salvar predições brutas
    with open(predicoes_path, "w", encoding="utf-8") as f:
        for pred in y_pred_str:
            f.write(f"{pred}\n")
    print(f"[OK] Predições salvas em '{predicoes_path}'")

    # Gerar e salvar Matriz de Confusão
    try:
        cm = confusion_matrix(y_test, y_pred_str, labels=label_encoder.classes_)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format="d")
        plt.title(
            f"Matriz de Confusão - {model_name} (Acc: {acc * 100:.1f}% | F1-Macro: {f1_mac *100:.1f}%)",
            fontsize=14, pad=15,
        )
        plt.xlabel("Gesto Predito", fontsize=12)
        plt.ylabel("Gesto Real", fontsize=12)
        plt.tight_layout()
        plt.savefig(matriz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"[AVISO] Não foi possível salvar gráfico da matriz de confusão: {e}")



if __name__ == "__main__":
    # TODO: Adicionar os outros modelos para o treinamento
    models = [
        # Modelo, Hiperparametros, Path binário, Usar Augmentation?
        ('lstm', PARAM_GRID_LSTM, LSTM_PATH, False),
        ('lstm', PARAM_GRID_LSTM, LSTM_PATH_AUG, True),
        ('kmeans', PARAM_GRID_KMEANS, KMEANS_PATH, False),
    ]

    X_train, y_train, groups_train = import_from_csv(DATASET_TREINO_CSV)
    X_test, y_test, groups_test = import_from_csv(DATASET_TESTE_CSV)


    X_train_reduzido, y_train_reduzido, groups_train_reduzido, X_val, y_val, groups_val = splitTrainValidation(
        x_fit=X_train, y_fit=y_train, groups_fit=groups_train
    )
    #X_aug, y_aug = gerar_amostras_aumentadas(X_train, y_train)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_reduzido_encoded = label_encoder.transform(y_train_reduzido)
    y_val_encoded = label_encoder.transform(y_val)
    num_labels = len(label_encoder.classes_)

    print("Shape Dados  de Treino: ", X_train.shape)
    print("Shape Labels de Treino: ", y_train.shape)
    print("Shape Grupos de Treino: ", groups_train.shape)

    print("Shape Dados  de Validação: ", X_val.shape)
    print("Shape Labels de Validação: ", y_val.shape)
    print("Shape Grupos de Validação: ", groups_val.shape)

    print("Shape Dados  de Teste: ", X_test.shape)
    print("Shape Labels de Teste: ", y_test.shape)
    print("Shape Grupos de Teste: ", groups_test.shape)

    # TODO Adicionar o Data Augmentation aqui para validar os modelos
    
    input_shape = (X_train.shape[1], X_train.shape[2])

    # F1-Score Macro como métrica para
    # garantir os melhores parâmetros
    # pois realiza a análise de acerto das
    # classes com o mesmo peso para todas as classes
    scoring_method = "f1_macro"
    for model_name, params, path_model, usar_augmentation in models:
        match(model_name):
            case "lstm":
                # Configuração de Early Stopping para evitar
                # a passagem por todas as 100 épocas em todos
                # os treinamentos
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )
                model = KerasClassifier(
                    model=create_lstm_model,
                    model__input_shape=input_shape,
                    model__num_classes=num_labels,
                    verbose=0,
                    callbacks=[early_stopping]
                )
            case "kmeans":
                model = criar_pipeline_kmeans()
            case default:
                pass

        eh_modelo_keras = model_name == "lstm"

        # LSTM precisa reservar dados de validação pro EarlyStopping; KMeans/RF não usa
        # validação separada, então treina com o dataset de treino completo.
        if eh_modelo_keras:
            x_fit, y_fit, groups_fit = X_train_reduzido, y_train_reduzido_encoded, groups_train_reduzido
        else:
            x_fit, y_fit, groups_fit = X_train, y_train_encoded, groups_train

        # Aplica o Grid Search no modelo
        #if usar_augmentation:
        #    x_fit, y_fit = X_aug, label_encoder.transform(y_aug)
        #else:

        best_model, best_params, best_score = applyGridSearch(
            model_used=model,
            params=params,
            scoring_method=scoring_method,
            X_fit=x_fit,
            y_fit=y_fit,
            groups=groups_fit,
            X_val=X_val,
            y_val=y_val_encoded,
            usa_validation_data=eh_modelo_keras,
        )

        if model_name == "lstm":
            predicoes_path, matriz_path = PREDICOES_LSTM_PATH, MATRIZ_LSTM_PATH
        else:
            predicoes_path, matriz_path = PREDICOES_KMEANS_PATH, MATRIZ_KMEANS_PATH

        # Realiza o teste e grava métricas em arquivos
        avaliar_modelo_teste(
            model_name=model_name,
            model=best_model,
            best_params=best_params,
            label_encoder=label_encoder,
            X_test=X_test,
            y_test=y_test,
            usa_predict_proba=eh_modelo_keras,
            predicoes_path=predicoes_path,
            matriz_path=matriz_path,
        )

        if eh_modelo_keras:
            best_model.model_.save(path_model)
        else:
            with open(path_model, "wb") as f:
                pickle.dump(best_model, f)
            print(f"[OK] Modelo {model_name} salvo em '{path_model}'")

    # Armazena o LabelEncoder em disco
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
