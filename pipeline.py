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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from feature_extraction import extract_features_from_directory, import_from_csv
from landmark_augmentation import gerar_amostras_aumentadas

# --- CONFIGURAÇÕES DO PIPELINE ---
K_FOLDS = 5
EPOCHS_POR_FOLD = 50
USAR_AUGMENTATION = True
N_AUMENTOS = 5
SEED = 42

# Grade de Hiperparâmetros para Busca (Grid Search)
PARAM_GRID = {
    "units_1": [32, 64],
    "units_2": [64, 128],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "batch_size": [8, 16],
}


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
    model = Sequential(
        [
            LSTM(
                units_1,
                return_sequences=True,
                input_shape=input_shape,
                recurrent_dropout=0.1,
            ),
            Dropout(dropout),
            LSTM(units_2, return_sequences=False, recurrent_dropout=0.1),
            Dropout(dropout),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def carregar_dados(
    dataset_root="dataset/frames_treino", csv_path="dataset/dataset_completo_lstm.csv"
):
    """
    Carrega o dataset a partir do CSV pré-existente ou extrai diretamente dos diretórios de frames.
    """
    if os.path.exists(csv_path):
        print(f"[INFO] Carregando dataset pré-gerado de '{csv_path}'...")
        features, labels = import_from_csv(csv_path, mode="lstm")
    elif os.path.exists(dataset_root):
        print(f"[INFO] Extraindo features do diretório '{dataset_root}'...")
        features, labels = extract_features_from_directory(
            dataset_root_dir=dataset_root, mode="lstm", export_dataframe=True
        )
    else:
        raise FileNotFoundError(
            f"Nem o CSV '{csv_path}' nem o diretório '{dataset_root}' foram encontrados."
        )

    return np.array(features), np.array(labels)


def executar_grid_search_cv(X, y):
    """
    Executa o Grid Search combinado com Stratified K-Fold Cross-Validation.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    input_shape = (X.shape[1], X.shape[2])

    # Gerar todas as combinações de hiperparâmetros
    keys, values = zip(*PARAM_GRID.items())
    combinações = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("=" * 70)
    print("   PIPELINE LSTM: GRID SEARCH & K-FOLD CROSS-VALIDATION   ")
    print("=" * 70)
    print(f"Total de Amostras: {len(X)} | Classes: {num_classes}")
    print(f"Número de Folds (K): {K_FOLDS}")
    print(f"Total de Combinações no Grid Search: {len(combinações)}")
    print(f"Augmentation no Treino dos Folds: {'SIM' if USAR_AUGMENTATION else 'NÃO'}")
    print("=" * 70 + "\n")

    resultados = []
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for i, params in enumerate(combinações, 1):
        print(f"\n--- [Combinação {i}/{len(combinações)}] ---")
        print(f" Hiperparâmetros: {params}")

        fold_accuracies = []
        fold_f1_macros = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_str, y_val_str = y[train_idx], y[val_idx]

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
            y_train_cat = to_categorical(label_encoder.transform(y_train_str))
            y_val_cat = to_categorical(label_encoder.transform(y_val_str))

            # Pesos balanceados por classe
            classes_unicas = np.unique(y_train_str)
            pesos_array = compute_class_weight(
                "balanced", classes=classes_unicas, y=y_train_str
            )
            class_weight_dict = {
                int(label_encoder.transform([c])[0]): float(p)
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
                monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
            )

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
            y_val_pred_probs = model.predict(X_val_fold, verbose=0)
            y_val_pred_idx = np.argmax(y_val_pred_probs, axis=1)
            y_val_true_idx = label_encoder.transform(y_val_str)

            acc = accuracy_score(y_val_true_idx, y_val_pred_idx)
            f1_mac = f1_score(
                y_val_true_idx, y_val_pred_idx, average="macro", zero_division=0
            )

            fold_accuracies.append(acc)
            fold_f1_macros.append(f1_mac)

        acc_media = np.mean(fold_accuracies)
        acc_std = np.std(fold_accuracies)
        f1_media = np.mean(fold_f1_macros)
        f1_std = np.std(fold_f1_macros)

        print(f" -> Resultado Médio [{K_FOLDS} Folds]:")
        print(f"    - Acurácia Médias: {acc_media * 100:.2f}% (± {acc_std * 100:.2f}%)")
        print(f"    - F1-Score Macro:  {f1_media * 100:.2f}% (± {f1_std * 100:.2f}%)")

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
        by="f1_macro_mean", ascending=False
    )

    os.makedirs("outputs", exist_ok=True)
    output_csv = "outputs/grid_search_results.csv"
    df_resultados.to_csv(output_csv, index=False)
    print(f"\n[OK] Resultados completos salvos em '{output_csv}'")

    print("\n" + "=" * 70)
    print("   LEADERBOARD - TOP 3 MELHORES CONFIGURAÇÕES   ")
    print("=" * 70)
    for rank, (_, row) in enumerate(df_resultados.head(3).iterrows(), 1):
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


def treinar_modelo_final(X, y, melhor_config, label_encoder):
    """
    Treina o modelo final sobre TODO o dataset com os melhores hiperparâmetros encontrados.
    """
    print("\n" + "=" * 70)
    print("   TREINANDO MODELO FINAL COM OS MELHORES HIPERPARÂMETROS   ")
    print("=" * 70)

    num_classes = len(label_encoder.classes_)
    input_shape = (X.shape[1], X.shape[2])

    if USAR_AUGMENTATION:
        X_aug, y_aug = gerar_amostras_aumentadas(
            X.tolist(), y.tolist(), mode="lstm", n_aumentos=N_AUMENTOS
        )
        X_final = np.array(X_aug)
        y_final_str = np.array(y_aug)
    else:
        X_final = X
        y_final_str = y

    y_final_cat = to_categorical(label_encoder.transform(y_final_str))

    classes_unicas = np.unique(y_final_str)
    pesos_array = compute_class_weight(
        "balanced", classes=classes_unicas, y=y_final_str
    )
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

    print("Treinando o modelo final...")
    model.fit(
        X_final,
        y_final_cat,
        epochs=EPOCHS_POR_FOLD,
        batch_size=int(melhor_config["batch_size"]),
        class_weight=class_weight_dict,
        verbose=1,
    )

    os.makedirs("models", exist_ok=True)
    model_path = "models/lstm_sign_model.h5"
    encoder_path = "models/label_encoder.pkl"

    model.save(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\n[OK] Modelo LSTM final salvo em '{model_path}'")
    print(f"[OK] Label Encoder salvo em '{encoder_path}'")


if __name__ == "__main__":
    X, y = carregar_dados()
    melhor_config, label_encoder = executar_grid_search_cv(X, y)
    treinar_modelo_final(X, y, melhor_config, label_encoder)
