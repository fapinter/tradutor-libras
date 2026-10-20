"""
pipeline.py
===========
Pipeline completo de Treinamento, Otimização de Hiperparâmetros (Grid Search)
e Validação Cruzada K-Fold Estratificada para o modelo LSTM de Tradução de Libras.

Métricas de Avaliação:
- Acurácia (Accuracy)
- F1-Score Macro (F1-Macro)
"""

# Biblotecas externas utilizadas
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback

import sys
import gc
import joblib
import time
from tqdm.auto import tqdm

# Imports do próprio projeto
from feature_extraction import import_from_csv
from landmark_augmentation import gerar_amostras_aumentadas

from utils.tqdm import tqdm_joblib
from utils.utils import _salvar_log_treino
from utils.kmeans import criar_pipeline_kmeans
from utils.xgboost import (
    criar_pipeline_xgboost,
    criar_pipeline_rf
)
from utils.lstm_tcn import(
    create_lstm_model,
    create_tcn_model
)
from utils.constants import (
    ENCODER_PATH,
    MATRIZ_PATH,
    N_AUMENTOS,
    PARAM_GRID_KMEANS,
    PARAM_GRID_RF,
    PARAM_GRID_LSTM,
    PARAM_GRID_TCN,
    PARAM_GRID_XGBOOST,
    RESULTS_PATH,
    KERAS_PATH,
    PICKLE_PATH,
    DATASET_TREINO_15_CSV,
    DATASET_TESTE_15_CSV,
    DATASET_TREINO_MINDS_CSV,
    DATASET_TESTE_MINDS_CSV,
    LOSS_PATH
)

class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            gc.collect()

    def on_train_end(self, logs=None):
        tf.contrib.keras.backend.clear_session()
        gc.collect()


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
    
    

def applyGridSearch(
    model_used, params, scoring_method,
    X_fit, y_fit, groups,
    X_val, y_val, usa_validation_data=True
):
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

    if usa_validation_data:
        n_jobs = 2
    else:
        n_jobs = -1

    grid = GridSearchCV(
        estimator=model_used,
        param_grid=params,
        cv=cv,
        scoring=scoring_method,
        n_jobs=n_jobs,
        verbose=1
    )
    total_fits = len(list(ParameterGrid(params))) * 5

    # Barra de progresso do treinamento
    with tqdm_joblib(tqdm(desc="Treinando GridSearchCV", total=total_fits)):
        grid.fit(X_fit, y_fit, groups=groups)
    return grid.best_estimator_, grid.best_params_, grid.best_score_



def avaliar_modelo_teste(
    model_name, model, best_params,
    label_encoder, X_test, y_test,
    n_amostras_treino, n_aumentos, augmentation,
    time, usa_predict_proba=True,
):
    """
    Realiza o Teste do Modelo, coleta as Métricas (F1-Score Macro e Acurácia Geral)
    e gera a Matriz de Confusão da predição do modelo
    """
    matriz_path = MATRIZ_PATH % model_name
    results_path = RESULTS_PATH % model_name


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
    minutos_treino = time/60
    print(f"Tempo total de Treino: {minutos_treino:.4f} minutos | {time/3600:.4f} horas")

    if usa_predict_proba:
        epoca_parada = len(model.history_["loss"])
    else:
        epoca_parada = None
    _salvar_log_treino(
        model_name, acc, f1_mac,
        n_amostras_treino, augmentation, n_aumentos, 
        str(best_params.items()), minutos_treino, epoca_parada
    )
    # Salva resultados em outputs/
    with open(results_path, 'w') as file_:
        file_.write(f"Total de Amostras de Teste: {len(y_test)}\n")
        file_.write(f"Acurácia no Teste: {acc * 100:.2f}%\n")
        file_.write(f"F1-Score Macro:             {f1_mac * 100:.2f}%\n")
        file_.write(f"Tempo total de Treino: {time/60:.4f} minutos | {time/3600:.4f} horas\n")
        file_.write(f"Época parada: {epoca_parada}")
        file_.write(f"Hiperparametros do {model_name}: {best_params.items()}")
        file_.write("\nRelatório de Classificação por Classe:\n")
        file_.write(report_text)

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

    try:
        if epoca_parada:
            loss_path = LOSS_PATH % model_name
            history = model.history_
            
            # Identificar chaves de acurácia compatíveis com a versão do TensorFlow
            acc_key = 'accuracy' if 'accuracy' in history else 'acc'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
            # Definir o eixo X (épocas)
            epocas = range(1, len(history[acc_key]) + 1)
            ultima_epoca = len(epocas)
            
            # Criar o gráfico
            plt.figure(figsize=(10, 6))
            
            plt.plot(epocas, history[acc_key], label='Acurácia no Treino', marker='o', color='#1f77b4', linewidth=2)
            plt.plot(epocas, history[val_acc_key], label='Acurácia na Validação', marker='s', color='#ff7f0e', linewidth=2)
            
            # Adicionar uma linha vertical marcando onde o Early Stopping atuou
            plt.axvline(
                x=ultima_epoca, color='red', linestyle=':', 
                label=f'Early Stopping (Época {ultima_epoca})'
            )
            
            # Estilização
            plt.title('Análise de Overfitting: Treino vs Validação (Melhor Modelo)', fontsize=14)
            plt.xlabel('Épocas', fontsize=12)
            plt.ylabel('Acurácia', fontsize=12)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(loss_path, dpi=150, bbox_inches="tight")

    except Exception as e:
        print(f"[AVISO] Não foi possível salvar gráfico da acurácia da rede: {e}")


if __name__ == "__main__":

    models = [
        # Modelo, Hiperparametros, Path binário, Usar Augmentation
        #('lstm:minds_aug', PARAM_GRID_LSTM, True),
        #('lstm:minds', PARAM_GRID_LSTM,  False),
        #('tcn:aug_15', PARAM_GRID_TCN,  True),
        #('tcn:15', PARAM_GRID_TCN,  False),
        #('xgboost:minds_aug', PARAM_GRID_XGBOOST, True),
        #('xgboost:minds', PARAM_GRID_XGBOOST, False),
        ('rf:15_aug', PARAM_GRID_RF,  True),
        ('rf:15', PARAM_GRID_RF,  False),
    ]
    has_network = False
    has_classifier = False
    for model_name,_,_ in models:
        if model_name.startswith('lstm') or model_name.startswith('tcn'):
            has_network = True
        elif model_name.startswith('kmeans') or model_name.startswith('xgboost'):
            has_kmeans = True

        if has_network and has_classifier:
            print('[ERRO] Não é permitido o treinamento de Kmeans e LSTM juntos\nFaça o treinamento em execuções diferentes')
            sys.exit(1) 
    if has_network:
        TREINAR_REDE = True
    else:
        TREINAR_REDE = False
    # Dataset é definido aqui, não possui opção de mudar
    
    # Coleta dos dados no dtype=np.float32
    X_train, y_train, groups_train = import_from_csv(DATASET_TREINO_15_CSV)
    X_test, y_test, groups_test = import_from_csv(DATASET_TESTE_15_CSV)

    # Separação dos dados de treino e validação
    X_train_reduzido, y_train_reduzido, groups_train_reduzido, X_val, y_val, groups_val = splitTrainValidation(
        x_fit=X_train, y_fit=y_train, groups_fit=groups_train
    )

    label_encoder = LabelEncoder()
    # Data Augmentation offline (antes do treinamento)
    # Para a LSTM e TCN, os dados de validação são separados para o Early Stopping
    if TREINAR_REDE:
        X_aug, y_aug, group_aug = gerar_amostras_aumentadas(
            X_train_reduzido.tolist(), y_train_reduzido.tolist(), groups_train_reduzido.tolist(), n_aumentos=N_AUMENTOS
        )
        label_encoder.fit_transform(y_train_reduzido)
    else:
        X_aug, y_aug, group_aug = gerar_amostras_aumentadas(
            X_train.tolist(), y_train.tolist(), groups_train.tolist(), n_aumentos=N_AUMENTOS
        )
        label_encoder.fit_transform(y_train)

    y_train_reduzido_encoded = label_encoder.transform(y_train_reduzido)
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    num_labels = len(label_encoder.classes_)

    # F1-Score Macro como métrica para
    # garantir os melhores parâmetros
    # pois realiza a análise de acerto das
    # classes com o mesmo peso para todas as classes
    scoring_method = "f1_macro"
    for model_name, params, usar_augmentation in models:
        # Redução do Uso de RAM
        tf.keras.backend.clear_session()
        gc.collect()

        if usar_augmentation:
            x_fit, y_fit, group_fit = X_aug, label_encoder.transform(y_aug), group_aug
        else:
            if TREINAR_REDE:
                x_fit, y_fit, group_fit = X_train_reduzido, y_train_reduzido_encoded, groups_train_reduzido
            else:
                x_fit, y_fit, group_fit = X_train, y_train_encoded, groups_train
        
        input_shape = (x_fit.shape[1], x_fit.shape[2])
        print("Shape fit: ", x_fit.shape, y_fit.shape, group_fit.shape)
        print("dtype X fit: ", x_fit.dtype)
        model_to_create = model_name.split(':')[0]
        match(model_to_create):
            case "lstm":
                # Configuração de Early Stopping para evitar
                # a passagem por todas as 100 épocas em todos
                # os treinamentos
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )
                cleanup_cb = MemoryCleanupCallback()
                model = KerasClassifier(
                    model=create_lstm_model,
                    model__input_shape=input_shape,
                    model__num_classes=num_labels,
                    verbose=0,
                    callbacks=[early_stopping, cleanup_cb]
                )
            case "kmeans":
                model = criar_pipeline_kmeans()
            case "rf":
                model = criar_pipeline_rf()
            case "xgboost":
                model = criar_pipeline_xgboost()    
            case "tcn":
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )
                cleanup_cb = MemoryCleanupCallback()
                model = KerasClassifier(
                    model=create_tcn_model,
                    model__input_shape=input_shape,
                    model__num_classes=num_labels,
                    verbose=0,
                    callbacks=[early_stopping, cleanup_cb]
                )    
            case _:
                pass

        eh_modelo_keras = model_to_create in ("lstm", "tcn")
        start_time = time.perf_counter()
        # Aplica o Grid Search no modelo
        best_model, best_params, best_score = applyGridSearch(
            model_used=model,
            params=params,
            scoring_method=scoring_method,
            X_fit=x_fit,
            y_fit=y_fit,
            groups=group_fit,
            X_val=X_val,
            y_val=y_val_encoded,
            usa_validation_data=eh_modelo_keras,
        )
        end_time = time.perf_counter() - start_time
        # Realiza o teste e grava métricas em arquivos
        model_name = model_name.replace(':', '_')
        avaliar_modelo_teste(
            model_name=model_name,
            model=best_model,
            best_params=best_params,
            label_encoder=label_encoder,
            X_test=X_test,
            y_test=y_test,
            n_amostras_treino=x_fit.shape[0],
            n_aumentos=N_AUMENTOS,
            augmentation=usar_augmentation,
            usa_predict_proba=eh_modelo_keras,
            time=end_time
        )
        if eh_modelo_keras:
            best_model.model_.save(KERAS_PATH % model_name)
            print(f'[OK] Modelo {model_name} salvo em "{KERAS_PATH % model_name}"')
        else:
            joblib.dump(best_model, PICKLE_PATH % model_name, compress=3)
            print(f'[OK] Modelo {model_name} salvo em "{PICKLE_PATH % model_name}"')



    # Armazena o LabelEncoder em disco
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
