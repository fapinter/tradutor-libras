"""
Pipeline completo de Treinamento, Otimização de Hiperparâmetros (Grid Search)
e Validação Cruzada K-Fold Estratificada.

Métricas de Avaliação:
- F1-Score Macro (F1-Macro)
- Acurácia (Accuracy)
"""

# Biblotecas externas utilizadas
import csv
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
from utils.utils import (
    _salvar_log_treino,
    eh_melhor_treino,
    interprete_do_video,
    salvar_melhores_metricas
)
from utils.kmeans import criar_pipeline_kmeans
from utils.xgboost import (
    criar_pipeline_xgboost,
    criar_pipeline_rf
)
from utils.redes import(
    create_lstm_model,
    create_lstm_atencao_model,
    create_bilstm_model,
    create_bilstm_atencao_model,
    create_tcn_model,
    create_tcn_atencao_model,
)
from utils.constants import (
    ENCODER_PATH,
    MATRIZ_PATH,
    RESULTS_PATH,
    KERAS_PATH,
    PICKLE_PATH,
    ACC_PATH,
    LOSS_PATH,
    PREDICOES_PATH,
    N_AUMENTOS,
    INTERPRETES_VALIDACAO,
    EARLY_STOPPING_PATIENCE,
    PARAM_GRID_KMEANS,
    PARAM_GRID_RF,
    PARAM_GRID_LSTM,
    PARAM_GRID_TCN,
    PARAM_GRID_XGBOOST,
    DATASET_TREINO_CSV,
    DATASET_TESTE_CSV,
    DATASET_TREINO_HOLISTIC_CSV,
    DATASET_TESTE_HOLISTIC_CSV,
    DATASET_TREINO_15_CSV,
    DATASET_TESTE_15_CSV,
    DATASET_TREINO_MINDS_CSV,
    DATASET_TESTE_MINDS_CSV,
    DATASET_TREINO_MINDS_CROP_CSV,
    DATASET_TESTE_MINDS_CROP_CSV,
)


class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            gc.collect()

    def on_train_end(self, logs=None):
        tf.keras.backend.clear_session()
        gc.collect()


class F1MacroValidacao(Callback):
    """
    Calcula o F1-Macro na validação ao fim de cada época e o registra nos logs como 'val_f1_macro',
    para o EarlyStopping monitorá-lo (o Keras não calcula F1). Deve vir antes do EarlyStopping
    na lista de callbacks.
    """
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0, batch_size=256), axis=1)
        logs["val_f1_macro"] = f1_score(self.y_val, y_pred, average="macro", zero_division=0)


def criar_callbacks_rede(X_val, y_val):
    """
    Callbacks do treinamento das redes (LSTM e TCN): Early Stopping pelo F1-Macro da validação
    (a mesma métrica do GridSearch), restaurando os pesos da melhor época. A val_loss não é usada
    porque atinge o mínimo cedo (~época 5) e sobe enquanto o F1 da validação e do teste ainda melhoram.
    """
    early_stopping = EarlyStopping(
        monitor="val_f1_macro",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    return [F1MacroValidacao(X_val, y_val), early_stopping, MemoryCleanupCallback()]


def splitTrainValidation(x_fit, y_fit, groups_fit):
    """
    Separa dados de treino dos dados de validação para aplicar
    o EarlyStopping no GridSearch corretamente.
    A validação recebe os intérpretes de INTERPRETES_VALIDACAO (groups_fit = intérprete),
    sem augmentation; os demais ficam no treino.
    """
    eh_validacao = np.isin(groups_fit, INTERPRETES_VALIDACAO)
    if set(groups_fit[eh_validacao].tolist()) != set(INTERPRETES_VALIDACAO):
        raise ValueError(f'Intérpretes de validação {INTERPRETES_VALIDACAO} ausentes nos grupos do treino')
    train_idx, val_idx = np.where(~eh_validacao)[0], np.where(eh_validacao)[0]

    x_train = x_fit[train_idx]
    y_train = y_fit[train_idx]
    groups_train = groups_fit[train_idx]

    x_val = x_fit[val_idx]
    y_val = y_fit[val_idx]
    groups_val = groups_fit[val_idx]

    overlap = set(groups_train).intersection(set(groups_val))
    if overlap:
        raise ValueError(f'Overlap entre grupos de Treino e Validação: {overlap}')
    print(f"Grupos no treino: {sorted(set(groups_train.tolist()))} | na validação: {sorted(set(groups_val.tolist()))}")

    return x_train, y_train, groups_train, x_val, y_val, groups_val


def applyGridSearch(
    model_used, params, scoring_method,
    X_fit, y_fit, groups,
    X_val, y_val, usa_validation_data=True,
    random_state=None
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
        random_state=random_state
    )

    if usa_validation_data:
        n_jobs =-1
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
    label_encoder, X_test, y_test, groups_test, sample_ids_test,
    n_amostras_treino, n_aumentos, augmentation,
    time, usa_predict_proba=True,
    X_val=None, y_val=None, f1_validacao_cruzada=None,
):
    """
    Realiza o Teste do Modelo, coleta as Métricas (F1-Score Macro e Acurácia Geral)
    e gera a Matriz de Confusão da predição do modelo, o CSV com a predição de cada
    amostra do teste e, para redes neurais, os gráficos de acurácia e loss por época.

    X_val, y_val (encoded): conjunto de validação (intérpretes de INTERPRETES_VALIDACAO), avaliado
    apenas nas redes neurais; os demais modelos treinam com todos os intérpretes e não têm validação
    separada. f1_validacao_cruzada: melhor F1-Macro médio dos folds do GridSearch (best_score_).

    Os arquivos em outputs/ só são sobrescritos se o treino superar o melhor treino salvo
    em best/<model_name>.txt (ver eh_melhor_treino). O log em logs/ é gravado sempre.

    Retorna se o treino é o novo melhor, o F1-Macro e a acurácia (em %).
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

    # Métricas na validação, com os pesos restaurados pelo Early Stopping
    acc_val = f1_val = None
    if X_val is not None:
        if usa_predict_proba:
            y_val_pred = np.argmax(model.predict_proba(X_val, verbose=0), axis=1)
        else:
            y_val_pred = model.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)
        f1_val = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"Validação | Acurácia: {acc_val * 100:.2f}% | F1-Macro: {f1_val * 100:.2f}%")
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

    # Mantém os resultados do melhor treino já salvo caso o atual não o supere
    f1_pct, acc_pct = f1_mac * 100, acc * 100
    if not eh_melhor_treino(model_name, f1_pct, acc_pct):
        print(f"[BEST] Treino não superou o melhor de {model_name}: modelo e outputs mantidos")
        return False, f1_pct, acc_pct

    # Salva resultados em outputs/
    with open(results_path, 'w', encoding='utf-8') as file_:
        file_.write(f"Total de Amostras de Teste: {len(y_test)}\n")
        file_.write(f"Acurácia no Teste: {acc * 100:.2f}%\n")
        file_.write(f"F1-Score Macro:             {f1_mac * 100:.2f}%\n")
        if X_val is not None:
            file_.write(f"Total de Amostras de Validação: {len(y_val)} (intérpretes {', '.join(map(str, INTERPRETES_VALIDACAO))})\n")
            file_.write(f"Acurácia na Validação: {acc_val * 100:.2f}%\n")
            file_.write(f"F1-Score Macro na Validação: {f1_val * 100:.2f}%\n")
        if f1_validacao_cruzada is not None:
            file_.write(f"F1-Score Macro médio na validação cruzada do GridSearch: {f1_validacao_cruzada * 100:.2f}%\n")
        file_.write(f"Tempo total de Treino: {time/60:.4f} minutos | {time/3600:.4f} horas\n")
        file_.write(f"Época parada: {epoca_parada}\n")
        file_.write(f"Hiperparametros do {model_name}: {best_params.items()}")
        file_.write("\nRelatório de Classificação por Classe:\n")
        file_.write(report_text)

    # Salva a predição de cada amostra do teste em outputs/
    # id = {nome_gesto}_{id_video}_{sample_idx}
    predicoes_path = PREDICOES_PATH % model_name
    with open(predicoes_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'predicao', 'diagnostico'])
        for video_id, sample_idx, gesto_real, gesto_previsto in zip(groups_test, sample_ids_test, y_test, y_pred_str):
            diagnostico = 'certo' if gesto_previsto == gesto_real else 'errado'
            writer.writerow([f'{video_id}_{sample_idx}', gesto_previsto, diagnostico])

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
            acc_path = ACC_PATH % model_name
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
            # F1-Macro da validação, monitorado pelo Early Stopping
            if 'val_f1_macro' in history:
                plt.plot(epocas, history['val_f1_macro'], label='F1-Macro na Validação', marker='^', color='#2ca02c', linewidth=2)
                melhor_epoca = int(np.argmax(history['val_f1_macro'])) + 1
                plt.axvline(
                    x=melhor_epoca, color='green', linestyle='--',
                    label=f'Maior F1-Macro na Validação (Época {melhor_epoca})'
                )

            # Adicionar uma linha vertical marcando onde o Early Stopping atuou
            plt.axvline(
                x=ultima_epoca, color='red', linestyle=':',
                label=f'Early Stopping (Época {ultima_epoca})'
            )

            # Estilização
            plt.title('Análise de Overfitting: Treino vs Validação (Melhor Modelo)', fontsize=14)
            plt.xlabel('Épocas', fontsize=12)
            plt.ylabel('Acurácia / F1-Macro', fontsize=12)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(acc_path, dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"[AVISO] Não foi possível salvar gráfico da acurácia da rede: {e}")

    try:
        if epoca_parada:
            loss_path = LOSS_PATH % model_name
            history = model.history_

            epocas = range(1, len(history['loss']) + 1)
            ultima_epoca = len(epocas)
            # Época restaurada pelo EarlyStopping (restore_best_weights=True), que monitora o F1-Macro da validação
            melhor_epoca = int(np.argmax(history['val_f1_macro'])) + 1

            plt.figure(figsize=(10, 6))

            plt.plot(epocas, history['loss'], label='Loss no Treino', marker='o', color='#1f77b4', linewidth=2)
            plt.plot(epocas, history['val_loss'], label='Loss na Validação', marker='s', color='#ff7f0e', linewidth=2)

            plt.axvline(
                x=melhor_epoca, color='green', linestyle='--',
                label=f'Maior F1-Macro na Validação (Época {melhor_epoca})'
            )
            plt.axvline(
                x=ultima_epoca, color='red', linestyle=':',
                label=f'Early Stopping (Época {ultima_epoca})'
            )

            plt.title('Análise de Overfitting: Loss Treino vs Validação (Melhor Modelo)', fontsize=14)
            plt.xlabel('Épocas', fontsize=12)
            plt.ylabel('Loss (Sparse Categorical Crossentropy)', fontsize=12)
            plt.legend(loc='upper right', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(loss_path, dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"[AVISO] Não foi possível salvar gráfico da loss da rede: {e}")

    return True, f1_pct, acc_pct


if __name__ == "__main__":

    models = [
        # Modelo, Hiperparametros, Path binário, Usar Augmentation, Número de Execuções
        # Modelos com atenção temporal aprendida (sufixo _at)
        ('lstm_at:att', PARAM_GRID_LSTM, False, 4),
        ('lstm_at:att_aug', PARAM_GRID_LSTM, True, 4),
        ('bilstm_at:att', PARAM_GRID_LSTM, False, 4),
        ('bilstm_at:att_aug', PARAM_GRID_LSTM, True, 4),
        ('tcn_at:att', PARAM_GRID_TCN, False, 3),
        ('tcn_at:att_aug', PARAM_GRID_TCN, True, 3),
        #('lstm:final', PARAM_GRID_LSTM,  False, 4),
        #('lstm:final_aug', PARAM_GRID_LSTM, True, 4),
        #('bilstm:final', PARAM_GRID_LSTM, False, 4),
        #('bilstm:final_aug', PARAM_GRID_LSTM, True, 4),
        #('tcn:final_aug', PARAM_GRID_TCN,  True, 3),
        #('tcn:final', PARAM_GRID_TCN,  False, 3),
        #('rf:pose_aug', PARAM_GRID_RF,  True, 4),
        #('rf:pose', PARAM_GRID_RF,  False, 4),
        #('kmeans:final_aug', PARAM_GRID_KMEANS, True, 4),
        #('kmeans:final', PARAM_GRID_KMEANS, False, 4),
        #('xgboost:final_aug', PARAM_GRID_XGBOOST, True, 4),
        #('xgboost:final', PARAM_GRID_XGBOOST, False, 4),

    ]
    has_network = False
    has_classifier = False
    for model_name, _, _, _ in models:
        if model_name.startswith('lstm') or model_name.startswith('bilstm') or model_name.startswith('tcn'):
            has_network = True
        elif model_name.startswith('kmeans') or model_name.startswith('xgboost') or model_name.startswith('rf'):
            has_classifier = True

        if has_network and has_classifier:
            print('[ERRO] Não é permitido o treinamento de Kmeans e LSTM juntos\nFaça o treinamento em execuções diferentes')
            sys.exit(1)
    if has_network:
        TREINAR_REDE = True
    else:
        TREINAR_REDE = False
    # Dataset é definido aqui, não possui opção de mudar

    # Coleta dos dados no dtype=np.float32
    X_train, y_train, groups_train = import_from_csv(DATASET_TREINO_HOLISTIC_CSV)
    X_test, y_test, groups_test, sample_ids_test = import_from_csv(DATASET_TESTE_HOLISTIC_CSV, retornar_sample_idx=True)

    # Grupos do treino passam a ser o intérprete (e não o vídeo), garantindo que validação
    # e folds do GridSearch tenham apenas intérpretes não vistos no treino
    groups_train = np.array([interprete_do_video(video_id) for video_id in groups_train])

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
    for model_name, params, usar_augmentation, n_execucoes in models:
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
        eh_modelo_keras = model_to_create in ("lstm", "lstm_at", "bilstm", "bilstm_at", "tcn", "tcn_at")
        for exec in range(n_execucoes):
            tf.keras.backend.clear_session()
            gc.collect()

            seed_exec = 42 + exec
            print(f"\n[{model_to_create}] Na execução {exec+1}/{n_execucoes}")
            match(model_to_create):
                case "lstm":
                    # Early Stopping pelo F1-Macro da validação (ver criar_callbacks_rede)
                    model = KerasClassifier(
                        model=create_lstm_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case "lstm_at":
                    model = KerasClassifier(
                        model=create_lstm_atencao_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case "bilstm":
                    model = KerasClassifier(
                        model=create_bilstm_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case "bilstm_at":
                    model = KerasClassifier(
                        model=create_bilstm_atencao_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case "kmeans":
                    model = criar_pipeline_kmeans()
                case "rf":
                    model = criar_pipeline_rf(seed_exec)
                case "xgboost":
                    model = criar_pipeline_xgboost(seed_exec)
                case "tcn":
                    model = KerasClassifier(
                        model=create_tcn_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case "tcn_at":
                    model = KerasClassifier(
                        model=create_tcn_atencao_model,
                        model__input_shape=input_shape,
                        model__num_classes=num_labels,
                        verbose=0,
                        callbacks=criar_callbacks_rede(X_val, y_val_encoded)
                    )
                case _:
                    pass

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
                random_state=seed_exec
            )
            end_time = time.perf_counter() - start_time
            # Realiza o teste e grava métricas em arquivos
            model_name = model_name.replace(':', '_')
            melhor_treino, f1_pct, acc_pct = avaliar_modelo_teste(
                model_name=model_name,
                model=best_model,
                best_params=best_params,
                label_encoder=label_encoder,
                X_test=X_test,
                y_test=y_test,
                groups_test=groups_test,
                sample_ids_test=sample_ids_test,
                n_amostras_treino=x_fit.shape[0],
                n_aumentos=N_AUMENTOS,
                augmentation=usar_augmentation,
                usa_predict_proba=eh_modelo_keras,
                time=end_time,
                X_val=X_val if eh_modelo_keras else None,
                y_val=y_val_encoded if eh_modelo_keras else None,
                f1_validacao_cruzada=best_score,
            )
            # Substitui o modelo em models/ apenas quando o treino supera o melhor já salvo
            if melhor_treino:
                if eh_modelo_keras:
                    best_model.model_.save(KERAS_PATH % model_name)
                    print(f'[OK] Modelo {model_name} salvo em "{KERAS_PATH % model_name}"')
                else:
                    joblib.dump(best_model, PICKLE_PATH % model_name, compress=3)
                    print(f'[OK] Modelo {model_name} salvo em "{PICKLE_PATH % model_name}"')
                # Métricas gravadas após o modelo, para best/ não apontar para um modelo que não foi salvo
                salvar_melhores_metricas(model_name, f1_pct, acc_pct)

    # Armazena o LabelEncoder em disco
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)