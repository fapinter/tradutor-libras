"""
Pipeline completo de Treinamento, Otimização de Hiperparâmetros (Grid Search)
e Validação Cruzada Leave-One-Group-Out (um intérprete por fold).

Fluxo das redes (LSTM, BiLSTM e TCN):
1. Os dados de treino são lidos do CSV e copiados; o augmentation é aplicado uma única vez na
   cópia (treino completo, 8 intérpretes).
2. A validação (INTERPRETES_VALIDACAO, sem augmentation) é separada dos dados originais e o
   split retorna os índices das amostras de treino.
3. O treino reduzido com augmentation é selecionado do treino completo com augmentation pelos
   índices de origem das amostras geradas.
4. GridSearch no treino reduzido: folds LeaveOneGroupOut nos intérpretes do treino, com Early
   Stopping na validação. Registra a melhor época de cada fold.
5. O modelo final é retreinado no treino completo com os melhores hiperparâmetros, por
   EPOCAS_TREINO_FINAL épocas, sem Early Stopping.

Os quatro conjuntos de treino (completo e reduzido, com e sem augmentation) são sempre criados;
USAR_AUGMENTATION de cada modelo escolhe entre as versões com e sem augmentation.

Os demais modelos (K-Means, RF e XGBoost) não usam validação: o GridSearch e o modelo final
usam diretamente o treino completo.

Métricas de Avaliação:
- F1-Score Macro (F1-Macro)
- Acurácia (Accuracy)
"""

# Biblotecas externas utilizadas
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
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
    N_AUMENTOS,
    INTERPRETES_VALIDACAO,
    EARLY_STOPPING_PATIENCE,
    EPOCAS_TREINO_FINAL,
    PARAM_GRID_KMEANS,
    PARAM_GRID_RF,
    PARAM_GRID_LSTM,
    PARAM_GRID_TCN,
    PARAM_GRID_XGBOOST,
    DATASET_TREINO_CSV,
    DATASET_TESTE_CSV,
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

    Retorna os índices das amostras de treino (para selecionar o treino reduzido, com ou sem
    augmentation) e os dados de validação.
    """
    eh_validacao = np.isin(groups_fit, INTERPRETES_VALIDACAO)
    if set(groups_fit[eh_validacao].tolist()) != set(INTERPRETES_VALIDACAO):
        raise ValueError(f'Intérpretes de validação {INTERPRETES_VALIDACAO} ausentes nos grupos do treino')
    train_idx, val_idx = np.where(~eh_validacao)[0], np.where(eh_validacao)[0]

    groups_train = groups_fit[train_idx]

    x_val = x_fit[val_idx]
    y_val = y_fit[val_idx]
    groups_val = groups_fit[val_idx]

    overlap = set(groups_train).intersection(set(groups_val))
    if overlap:
        raise ValueError(f'Overlap entre grupos de Treino e Validação: {overlap}')
    print(f"Grupos no treino: {sorted(set(groups_train.tolist()))} | na validação: {sorted(set(groups_val.tolist()))}")

    return train_idx, x_val, y_val, groups_val


def melhor_epoca_fold(estimator, X, y):
    """
    "Scorer" do GridSearch que retorna a época restaurada pelo Early Stopping no fold
    (maior F1-Macro da validação), para definir as épocas do treino final.
    """
    return int(np.argmax(estimator.history_["val_f1_macro"])) + 1


def applyGridSearch(
    model_used, params, scoring_method,
    X_fit, y_fit, groups,
    X_val, y_val, usa_validation_data=True,
):
    """
    Executa GridSearchCV para um dado modelo (pipeline) e espaço de parâmetros,
    com folds LeaveOneGroupOut (groups = intérprete, um intérprete por fold).
    Não há refit: o modelo final é treinado depois por treinar_modelo_final.

    Parâmetros:
        model_used              : Estimador base.
        params                  : dicionário de hiperparâmetros.
        scoring_method          : métrica de avaliação.
        X_fit, y_fit, groups    : dados de treino.
        X_val, y_val            : dados de validação
        usa_validation_data     : False pra modelos sklearn puros (não usam EarlyStopping)

    Retorna:
        best_params             : melhor combinação de hiperparâmetros.
        best_score              : melhor score médio dos folds.
        melhores_epocas         : época restaurada pelo Early Stopping em cada fold com os
                                  melhores hiperparâmetros (None para modelos sklearn).
    """

    if usa_validation_data:
        model_used.set_params(
            fit__validation_data=(X_val, y_val)
        )
        scoring = {scoring_method: scoring_method, "melhor_epoca": melhor_epoca_fold}
    else:
        scoring = {scoring_method: scoring_method}

    cv = LeaveOneGroupOut()

    grid = GridSearchCV(
        estimator=model_used,
        param_grid=params,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=-1,
        verbose=1
    )
    n_folds = cv.get_n_splits(groups=groups)
    total_fits = len(list(ParameterGrid(params))) * n_folds

    # Barra de progresso do treinamento
    with tqdm_joblib(tqdm(desc="Treinando GridSearchCV", total=total_fits)):
        grid.fit(X_fit, y_fit, groups=groups)

    resultados = grid.cv_results_
    melhor = int(np.argmin(resultados[f"rank_test_{scoring_method}"]))
    melhores_epocas = None
    if usa_validation_data:
        melhores_epocas = [int(resultados[f"split{i}_test_melhor_epoca"][melhor]) for i in range(n_folds)]
    return resultados["params"][melhor], resultados[f"mean_test_{scoring_method}"][melhor], melhores_epocas


def treinar_modelo_final(model_used, best_params, X_fit, y_fit, melhores_epocas=None):
    """
    Retreina o modelo com os melhores hiperparâmetros do GridSearch no treino inteiro
    (todos os intérpretes). Nas redes, não há validação nem Early Stopping: o treino dura
    EPOCAS_TREINO_FINAL épocas. melhores_epocas (melhores épocas dos folds) é apenas registrado.

    Retorna o modelo treinado e o número de épocas do treino (None para modelos sklearn).
    """
    model_final = clone(model_used).set_params(**best_params)
    epocas_final = None
    if melhores_epocas is not None:
        epocas_final = EPOCAS_TREINO_FINAL
        model_final.set_params(
            epochs=epocas_final,
            callbacks=[MemoryCleanupCallback()],
            fit__validation_data=None,
        )
        print(f"Treino final: {epocas_final} épocas (melhores épocas dos folds: {melhores_epocas})")
    model_final.fit(X_fit, y_fit)
    return model_final, epocas_final


def avaliar_modelo_teste(
    model_name, model, best_params,
    label_encoder, X_test, y_test,
    n_amostras_treino, n_aumentos, augmentation,
    time, usa_predict_proba=True,
    f1_validacao_cruzada=None, melhores_epocas=None,
):
    """
    Realiza o Teste do Modelo, coleta as Métricas (F1-Score Macro e Acurácia Geral)
    e gera a Matriz de Confusão da predição do modelo e, para redes neurais, os gráficos
    de acurácia e loss por época.

    O modelo final é treinado com todos os intérpretes (inclusive os da validação), então não há
    avaliação na validação. f1_validacao_cruzada: F1-Macro médio dos folds do GridSearch com os
    melhores hiperparâmetros. melhores_epocas: época restaurada pelo Early Stopping em cada fold
    (apenas redes), apenas registrada (o treino final dura EPOCAS_TREINO_FINAL épocas).

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
        if f1_validacao_cruzada is not None:
            file_.write(f"F1-Score Macro médio na validação cruzada do GridSearch: {f1_validacao_cruzada * 100:.2f}%\n")
        file_.write(f"Tempo total de Treino: {time/60:.4f} minutos | {time/3600:.4f} horas\n")
        if melhores_epocas is not None:
            file_.write(f"Melhor época de cada fold (Early Stopping na validação, intérpretes "
                        f"{', '.join(map(str, INTERPRETES_VALIDACAO))}): {melhores_epocas}\n")
            file_.write(f"Épocas do treino final (fixas, todos os intérpretes): {epoca_parada}\n")
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

    # Curvas do treino final das redes. Não há validação: o treino final usa todos os intérpretes,
    # e a análise de overfitting fica nas melhores épocas dos folds do GridSearch
    try:
        if epoca_parada:
            history = model.history_
            acc_key = 'accuracy' if 'accuracy' in history else 'acc'
            epocas = range(1, epoca_parada + 1)
            titulo = f'Treino final com todos os intérpretes ({epoca_parada} épocas, sem Early Stopping)'
            for path, chave, rotulo in (
                (ACC_PATH, acc_key, 'Acurácia'),
                (LOSS_PATH, 'loss', 'Loss (Sparse Categorical Crossentropy)'),
            ):
                plt.figure(figsize=(10, 6))
                plt.plot(epocas, history[chave], label=f'{rotulo.split(" (")[0]} no Treino', marker='o', color='#1f77b4', linewidth=2)
                plt.title(titulo, fontsize=14)
                plt.xlabel('Épocas', fontsize=12)
                plt.ylabel(rotulo, fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(path % model_name, dpi=150, bbox_inches="tight")
                plt.close()

    except Exception as e:
        print(f"[AVISO] Não foi possível salvar os gráficos de acurácia e loss da rede: {e}")

    return True, f1_pct, acc_pct


if __name__ == "__main__":

    models = [
        # Modelo, Hiperparametros, Path binário, Usar Augmentation, Número de Execuções
        # Modelos com atenção temporal aprendida (sufixo _at)
        #('bilstm_at:att_aug', PARAM_GRID_LSTM, True, 4),
        #('bilstm_at:att', PARAM_GRID_LSTM, False, 4),
        ('lstm_at:att_aug', PARAM_GRID_LSTM, True, 4),
        ('lstm_at:att', PARAM_GRID_LSTM, False, 4),
        #('tcn_at:att_aug', PARAM_GRID_TCN, True, 4),
        #('tcn_at:att', PARAM_GRID_TCN, False, 4),
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
    X_train, y_train, groups_train = import_from_csv(DATASET_TREINO_CSV)
    X_test, y_test, _ = import_from_csv(DATASET_TESTE_CSV)

    # Grupos do treino passam a ser o intérprete (e não o vídeo), garantindo que validação
    # e folds do GridSearch tenham apenas intérpretes não vistos no treino
    groups_train = np.array([interprete_do_video(video_id) for video_id in groups_train])

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    # Treino completo (todos os intérpretes), sem augmentation: cópia dos dados do CSV
    X_completo, y_completo, groups_completo = X_train.copy(), y_train.copy(), groups_train.copy()
    # Treino completo com Data Augmentation offline (antes do treinamento), gerado uma única vez.
    # origem_aug guarda o índice da amostra original de cada amostra gerada
    X_completo_aug, y_completo_aug, groups_completo_aug, origem_aug = gerar_amostras_aumentadas(
        X_completo.tolist(), y_completo.tolist(), groups_completo.tolist(),
        n_aumentos=N_AUMENTOS, retornar_origem=True,
    )

    # Separação da validação (sem augmentation) nos dados originais; o split retorna os índices do treino
    train_idx, X_val, y_val, groups_val = splitTrainValidation(
        x_fit=X_train, y_fit=y_train, groups_fit=groups_train
    )
    # Treino reduzido (sem os intérpretes de validação), sem augmentation
    X_reduzido, y_reduzido, groups_reduzido = X_completo[train_idx], y_completo[train_idx], groups_completo[train_idx]
    # Treino reduzido com augmentation: amostras do treino completo com augmentation geradas
    # a partir das amostras do treino reduzido
    eh_reduzido_aug = np.isin(origem_aug, train_idx)
    X_reduzido_aug = X_completo_aug[eh_reduzido_aug]
    y_reduzido_aug = y_completo_aug[eh_reduzido_aug]
    groups_reduzido_aug = groups_completo_aug[eh_reduzido_aug]

    # Conjuntos de treino por (tipo, usar_augmentation), com os rótulos codificados
    conjuntos_treino = {
        ("completo", False): (X_completo, label_encoder.transform(y_completo), groups_completo),
        ("completo", True): (X_completo_aug, label_encoder.transform(y_completo_aug), groups_completo_aug),
        ("reduzido", False): (X_reduzido, label_encoder.transform(y_reduzido), groups_reduzido),
        ("reduzido", True): (X_reduzido_aug, label_encoder.transform(y_reduzido_aug), groups_reduzido_aug),
    }
    for (tipo, com_aug), (x_conj, _, g_conj) in conjuntos_treino.items():
        print(f"Treino {tipo} {'com' if com_aug else 'sem'} augmentation: {x_conj.shape} | "
              f"intérpretes {sorted(set(g_conj.tolist()))}")

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

        # x_final, y_final: treino final com todos os intérpretes (treino completo)
        # x_fit, y_fit: GridSearch (treino reduzido nas redes, que usam a validação no Early Stopping,
        # e treino completo nos demais modelos)
        x_final, y_final, _ = conjuntos_treino["completo", usar_augmentation]
        x_fit, y_fit, group_fit = conjuntos_treino["reduzido" if TREINAR_REDE else "completo", usar_augmentation]

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
            best_params, best_score, melhores_epocas = applyGridSearch(
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
            # Retreina com os melhores hiperparâmetros no treino completo
            best_model, _ = treinar_modelo_final(model, best_params, x_final, y_final, melhores_epocas)
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
                n_amostras_treino=x_final.shape[0],
                n_aumentos=N_AUMENTOS,
                augmentation=usar_augmentation,
                usa_predict_proba=eh_modelo_keras,
                time=end_time,
                f1_validacao_cruzada=best_score,
                melhores_epocas=melhores_epocas,
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