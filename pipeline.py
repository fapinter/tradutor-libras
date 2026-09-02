"""
pipeline.py
===========
Pipeline completo de Treinamento, Otimização de Hiperparâmetros (Grid Search)
e Validação Cruzada K-Fold Estratificada para o modelo LSTM de Tradução de Libras.

Métricas de Avaliação:
- Acurácia (Accuracy)
- F1-Score Macro (F1-Macro)
"""

import csv
import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from feature_extraction import extract_features_from_directory, import_from_csv
from landmark_augmentation import gerar_amostras_aumentadas
from utils.constants import (
    DATASET_TESTE_CSV,
    DATASET_TREINO_CSV,
    ENCODER_PATH,
    EPOCHS_POR_FOLD,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    GRID_SEARCH_RESULTS_CSV,
    K_FOLDS,
    LOGS_DIR,
    MATRIZ_CONFUSAO_PNG,
    LSTM_PATH,
    MODELS_DIR,
    N_AUMENTOS,
    OUTPUTS_DIR,
    PARAM_GRID,
    PIPELINE_PATIENCE,
    PREDICOES_TESTE_PATH,
    RELATORIO_AVALIACAO_TESTE,
    SEED,
    USAR_AUGMENTATION,
)


def construir_modelo_lstm(
    input_shape,
    num_classes,
    units_1=64,
    units_2=128,
    dropout=0.3,
    learning_rate=0.001,
):
    """
    Constrói e compila o modelo LSTM com os hiperparâmetros fornecidos.
    """
    model = Sequential([
        LSTM(
            units_1,
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=0.1,
        ),
        Dropout(dropout),
        LSTM(units_2,
             return_sequences=False,
             recurrent_dropout=0.1),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def carregar_dados(
    dataset_root=FRAMES_TREINO_DIR,
    csv_path=DATASET_TREINO_CSV,
    return_groups=True,
):
    """
    Carrega o dataset de treino a partir do CSV pré-existente ou extrai diretamente dos diretórios de frames.
    Retorna features, labels e identificadores de grupos de vídeo.
    """
    if os.path.exists(csv_path):
        print(
            f"[INFO] Carregando dataset pré-gerado de '{csv_path}'..."
        )
        if return_groups:
            features, labels, groups = import_from_csv(
                csv_path, mode="lstm", return_groups=True)
            return np.array(features), np.array(
                labels), np.array(groups)
        else:
            features, labels = import_from_csv(
                csv_path, mode="lstm", return_groups=False)
            return np.array(features), np.array(labels)

    elif os.path.exists(dataset_root):
        print(
            f"[INFO] Extraindo features do diretório '{dataset_root}'..."
        )
        if return_groups:
            features, labels, groups = extract_features_from_directory(
                dataset_root_dir=dataset_root,
                mode="lstm",
                export_dataframe=True,
                output_csv_path=csv_path,
                return_groups=True,
            )
            return np.array(features), np.array(
                labels), np.array(groups)
        else:
            features, labels = extract_features_from_directory(
                dataset_root_dir=dataset_root,
                mode="lstm",
                export_dataframe=True,
                output_csv_path=csv_path,
                return_groups=False,
            )
            return np.array(features), np.array(labels)
    else:
        raise FileNotFoundError(
            f"Nem o CSV '{csv_path}' nem o diretório '{dataset_root}' foram encontrados."
        )


def executar_grid_search_cv(X, y, groups=None):
    """
    Executa o Grid Search combinado com Stratified K-Fold / StratifiedGroupKFold Cross-Validation.
    Garante isolamento completo contra Data Leakage:
      - Augmentation ocorre APENAS no conjunto de treino de cada fold.
      - Validação de cada fold permanece com dados autênticos não modificados.
      - Quando 'groups' está presente, amostras do mesmo vídeo permanecem no mesmo fold.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    input_shape = (X.shape[1], X.shape[2])

    # Gerar todas as combinações de hiperparâmetros
    keys, values = zip(*PARAM_GRID.items())
    combinações = [
        dict(zip(keys, v))
        for v in itertools.product(*values)
    ]

    # Escolha da estratégia de Cross-Validation
    tem_grupos_validos = (
        groups is not None and len(groups) == len(y)
        and len(np.unique(groups)) >= K_FOLDS
        and len(np.unique(groups)) < len(y))

    if tem_grupos_validos:
        splitter = StratifiedGroupKFold(n_splits=K_FOLDS,
                                        shuffle=True,
                                        random_state=SEED)
        splits = list(
            splitter.split(X, y_encoded, groups=groups))
        estrategia_desc = f"StratifiedGroupKFold ({len(np.unique(groups))} vídeos únicos agrupados)"
    else:
        splitter = StratifiedKFold(n_splits=K_FOLDS,
                                   shuffle=True,
                                   random_state=SEED)
        splits = list(splitter.split(X, y_encoded))
        estrategia_desc = "StratifiedKFold"

    print("=" * 70)
    print(
        "   PIPELINE LSTM: GRID SEARCH & K-FOLD CROSS-VALIDATION   "
    )
    print("=" * 70)
    print(
        f"Total de Amostras de Treino: {len(X)} | Classes: {num_classes}"
    )
    print(f"Número de Folds (K): {K_FOLDS}")
    print(f"Estratégia de Validação: {estrategia_desc}")
    print(
        f"Total de Combinações no Grid Search: {len(combinações)}"
    )
    print(
        f"Augmentation no Treino dos Folds: {'SIM' if USAR_AUGMENTATION else 'NÃO'}"
    )
    print("=" * 70 + "\n")

    resultados = []

    for i, params in enumerate(combinações, 1):
        print(
            f"\n--- [Combinação {i}/{len(combinações)}] ---"
        )
        print(f" Hiperparâmetros: {params}")

        fold_accuracies = []
        fold_f1_macros = []

        for fold, (train_idx,
                   val_idx) in enumerate(splits, 1):
            X_train_fold, X_val_fold = X[train_idx], X[
                val_idx]
            y_train_str, y_val_str = y[train_idx], y[
                val_idx]

            # Prevenção de Data Leakage: Augmentation APENAS no treino do fold
            if USAR_AUGMENTATION:
                X_train_aug, y_train_aug = gerar_amostras_aumentadas(
                    X_train_fold.tolist(),
                    y_train_str.tolist(),
                    mode="lstm",
                    n_aumentos=N_AUMENTOS,
                )
                X_train_fold = np.array(X_train_aug)
                y_train_str = np.array(y_train_aug)

            # One-hot encoding dos rótulos
            y_train_cat = to_categorical(
                label_encoder.transform(y_train_str))
            y_val_cat = to_categorical(
                label_encoder.transform(y_val_str))

            # Pesos balanceados por classe
            classes_unicas = np.unique(y_train_str)
            pesos_array = compute_class_weight(
                "balanced",
                classes=classes_unicas,
                y=y_train_str)
            class_weight_dict = {
                int(label_encoder.transform([c])[0]):
                float(p)
                for c, p in zip(classes_unicas, pesos_array)
            }

            model = construir_modelo_lstm(
                input_shape=input_shape,
                num_classes=num_classes,
                units_1=params["units_1"],
                units_2=params["units_2"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=PIPELINE_PATIENCE,
                restore_best_weights=True,
                verbose=0)

            model.fit(
                X_train_fold,
                y_train_cat,
                epochs=EPOCHS_POR_FOLD,
                batch_size=params["batch_size"],
                validation_data=(X_val_fold, y_val_cat),
                callbacks=[early_stop],
                class_weight=class_weight_dict,
                verbose=0,
            )

            # Avaliação no conjunto de validação do fold
            y_val_pred_probs = model.predict(X_val_fold,
                                             verbose=0)
            y_val_pred_idx = np.argmax(y_val_pred_probs,
                                       axis=1)
            y_val_true_idx = label_encoder.transform(
                y_val_str)

            acc = accuracy_score(y_val_true_idx,
                                 y_val_pred_idx)
            f1_mac = f1_score(y_val_true_idx,
                              y_val_pred_idx,
                              average="macro",
                              zero_division=0)

            fold_accuracies.append(acc)
            fold_f1_macros.append(f1_mac)

        acc_media = np.mean(fold_accuracies)
        acc_std = np.std(fold_accuracies)
        f1_media = np.mean(fold_f1_macros)
        f1_std = np.std(fold_f1_macros)

        print(f" -> Resultado Médio [{K_FOLDS} Folds]:")
        print(
            f"    - Acurácia Média: {acc_media * 100:.2f}% (± {acc_std * 100:.2f}%)"
        )
        print(
            f"    - F1-Score Macro:  {f1_media * 100:.2f}% (± {f1_std * 100:.2f}%)"
        )

        resultado_registro = {
            **params,
            "accuracy_mean": acc_media,
            "accuracy_std": acc_std,
            "f1_macro_mean": f1_media,
            "f1_macro_std": f1_std,
        }
        resultados.append(resultado_registro)

    # Converter para DataFrame e ordenar pelo F1-Score Macro
    df_resultados = pd.DataFrame(resultados).sort_values(
        by="f1_macro_mean", ascending=False)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_csv = GRID_SEARCH_RESULTS_CSV
    df_resultados.to_csv(output_csv, index=False)
    print(
        f"\n[OK] Resultados completos salvos em '{output_csv}'"
    )

    print("\n" + "=" * 70)
    print(
        "   LEADERBOARD - TOP 3 MELHORES CONFIGURAÇÕES   ")
    print("=" * 70)
    for rank, (_, row) in enumerate(
            df_resultados.head(3).iterrows(), 1):
        print(f"Rank {rank}:")
        print(
            f"  Params: units_1={int(row['units_1'])}, units_2={int(row['units_2'])}, "
            f"dropout={row['dropout']}, lr={row['learning_rate']}, batch={int(row['batch_size'])}"
        )
        print(
            f"  F1-Macro: {row['f1_macro_mean'] * 100:.2f}% | Acurácia: {row['accuracy_mean'] * 100:.2f}%"
        )
        print("-" * 70)

    melhor_config = df_resultados.iloc[0].to_dict()
    return melhor_config, label_encoder


def treinar_modelo_final(X, y, melhor_config,
                         label_encoder):
    """
    Treina o modelo final sobre TODO o dataset com os melhores hiperparâmetros encontrados.
    Retorna a instância do modelo treinado.
    """
    print("\n" + "=" * 70)
    print(
        "   TREINANDO MODELO FINAL COM OS MELHORES HIPERPARÂMETROS   "
    )
    print("=" * 70)

    num_classes = len(label_encoder.classes_)
    input_shape = (X.shape[1], X.shape[2])

    if USAR_AUGMENTATION:
        X_aug, y_aug = gerar_amostras_aumentadas(
            X.tolist(),
            y.tolist(),
            mode="lstm",
            n_aumentos=N_AUMENTOS)
        X_final = np.array(X_aug)
        y_final_str = np.array(y_aug)
    else:
        X_final = X
        y_final_str = y

    y_final_cat = to_categorical(
        label_encoder.transform(y_final_str))

    classes_unicas = np.unique(y_final_str)
    pesos_array = compute_class_weight(
        "balanced", classes=classes_unicas, y=y_final_str)
    class_weight_dict = {
        int(label_encoder.transform([c])[0]): float(p)
        for c, p in zip(classes_unicas, pesos_array)
    }

    model = construir_modelo_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
        units_1=int(melhor_config["units_1"]),
        units_2=int(melhor_config["units_2"]),
        dropout=float(melhor_config["dropout"]),
        learning_rate=float(melhor_config["learning_rate"]),
    )

    print(
        "Treinando o modelo final sobre todo o conjunto de treino..."
    )
    model.fit(
        X_final,
        y_final_cat,
        epochs=EPOCHS_POR_FOLD,
        batch_size=int(melhor_config["batch_size"]),
        class_weight=class_weight_dict,
        verbose=1,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    LSTM_PATH = LSTM_PATH
    encoder_path = ENCODER_PATH

    model.save(LSTM_PATH)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(
        f"\n[OK] Modelo LSTM final salvo em '{LSTM_PATH}'")
    print(f"[OK] Label Encoder salvo em '{encoder_path}'")

    return model


def avaliar_modelo_teste(
    model,
    label_encoder,
    test_csv_path=DATASET_TESTE_CSV,
    test_dataset_root=FRAMES_TESTE_DIR,
):
    """
    Avalia o modelo LSTM treinado sobre o conjunto de teste independente (Holdout Test Set).
    Gera métricas completas (Acurácia, F1-Macro, Precision, Recall), relatório de classificação
    e salva a Matriz de Confusão em 'logs/matriz_confusao_modelo_2.png'.
    """
    print("\n" + "=" * 70)
    print(
        "   AVALIAÇÃO FINAL NO CONJUNTO DE TESTE INDEPENDENTE (HOLDOUT)   "
    )
    print("=" * 70)

    if os.path.exists(test_csv_path):
        print(
            f"[INFO] Carregando dados de teste pré-gerados de '{test_csv_path}'..."
        )
        X_test, y_test = import_from_csv(test_csv_path,
                                         mode="lstm")
    elif os.path.exists(test_dataset_root):
        print(
            f"[INFO] Extraindo features de teste de '{test_dataset_root}'..."
        )
        X_test, y_test = extract_features_from_directory(
            dataset_root_dir=test_dataset_root,
            mode="lstm",
            export_dataframe=True,
            output_csv_path=test_csv_path,
        )
    else:
        print(
            f"⚠️  [AVISO] Conjunto de teste não encontrado em '{test_csv_path}' nem '{test_dataset_root}'."
        )
        print(
            "Dica: Execute a extração de frames de teste com 'data_preprocessing.py'."
        )
        return None

    X_test = np.array(X_test)
    y_test_str = np.array(y_test)

    if len(X_test) == 0:
        print(
            "⚠️  [AVISO] Nenhum dado de teste disponível para avaliação."
        )
        return None

    # Realiza as predições
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred_str = label_encoder.inverse_transform(y_pred_idx)

    # Filtra rótulos conhecidos no LabelEncoder
    y_test_encoded = []
    valid_mask = []
    for i, label in enumerate(y_test_str):
        if label in label_encoder.classes_:
            y_test_encoded.append(
                label_encoder.transform([label])[0])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            print(
                f"[AVISO] Classe desconhecida '{label}' no conjunto de teste."
            )

    y_test_encoded = np.array(y_test_encoded)
    y_test_valid = y_test_str[valid_mask]
    y_pred_valid = y_pred_str[valid_mask]
    y_pred_idx_valid = y_pred_idx[valid_mask]

    acc = accuracy_score(y_test_encoded, y_pred_idx_valid)
    f1_mac = f1_score(y_test_encoded,
                      y_pred_idx_valid,
                      average="macro",
                      zero_division=0)
    prec_mac = precision_score(y_test_encoded,
                               y_pred_idx_valid,
                               average="macro",
                               zero_division=0)
    rec_mac = recall_score(y_test_encoded,
                           y_pred_idx_valid,
                           average="macro",
                           zero_division=0)

    report_text = classification_report(y_test_valid,
                                        y_pred_valid,
                                        zero_division=0)

    print("\n" + "-" * 70)
    print(
        "                      MÉTRICAS DE TESTE FINAL                      "
    )
    print("-" * 70)
    print(
        f"Total de Amostras de Teste: {len(y_test_valid)}")
    print(f"Acurácia no Teste:          {acc * 100:.2f}%")
    print(
        f"F1-Score Macro:             {f1_mac * 100:.2f}%")
    print(
        f"Precisão Macro:             {prec_mac * 100:.2f}%"
    )
    print(
        f"Revocação (Recall) Macro:   {rec_mac * 100:.2f}%")
    print("-" * 70)
    print("\nRelatório de Classificação por Classe:")
    print(report_text)
    print("-" * 70)

    # Salvar relatório textual
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    relatorio_path = RELATORIO_AVALIACAO_TESTE
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(
            "RELATORIO DE AVALIACAO FINAL - MODELO LSTM (HOLDOUT TEST SET)\n"
        )
        f.write("=" * 70 + "\n\n")
        f.write(
            f"Total de Amostras:        {len(y_test_valid)}\n"
        )
        f.write(
            f"Acuracia:                 {acc * 100:.2f}%\n")
        f.write(
            f"F1-Score Macro:           {f1_mac * 100:.2f}%\n"
        )
        f.write(
            f"Precisao Macro:           {prec_mac * 100:.2f}%\n"
        )
        f.write(
            f"Revocacao (Recall) Macro: {rec_mac * 100:.2f}%\n\n"
        )
        f.write("Classificacao Detalhada por Gesto:\n")
        f.write(report_text + "\n")
    print(
        f"[OK] Relatório completo salvo em '{relatorio_path}'"
    )

    # Salvar predições brutas
    predicoes_path = PREDICOES_TESTE_PATH
    with open(predicoes_path, "w", encoding="utf-8") as f:
        for pred in y_pred_valid:
            f.write(f"{pred}\n")
    print(f"[OK] Predições salvas em '{predicoes_path}'")

    # Gerar e salvar Matriz de Confusão
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        cm = confusion_matrix(y_test_valid,
                              y_pred_valid,
                              labels=label_encoder.classes_)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_)
        disp.plot(cmap=plt.cm.Blues,
                  ax=ax,
                  xticks_rotation=45,
                  values_format="d")
        plt.title(
            f"Matriz de Confusão - LSTM (Teste Holdout | Acc: {acc * 100:.1f}%)",
            fontsize=14,
            pad=15,
        )
        plt.xlabel("Gesto Predito", fontsize=12)
        plt.ylabel("Gesto Real (Gabarito)", fontsize=12)
        plt.tight_layout()

        fig_path = MATRIZ_CONFUSAO_PNG
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"[OK] Matriz de Confusão salva em '{fig_path}'"
        )
    except Exception as e:
        print(
            f"[AVISO] Não foi possível salvar gráfico da matriz de confusão: {e}"
        )

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "precision_macro": prec_mac,
        "recall_macro": rec_mac,
    }


if __name__ == "__main__":
    X_train, y_train, groups = carregar_dados(
        return_groups=True)
    melhor_config, label_encoder = executar_grid_search_cv(
        X_train, y_train, groups=groups)
    model = treinar_modelo_final(X_train, y_train,
                                 melhor_config,
                                 label_encoder)
    avaliar_modelo_teste(model, label_encoder)
