import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from utils.constants import (
    ENCODER_PATH,
    KNN_PATH,
    LSTM_EPOCHS,
    LSTM_PATH,
    LSTM_PATIENCE,
    N_AUMENTOS,
    N_CLUSTERS,
    SEED,
)


def train_lstm(
    features,
    labels,
    LSTM_PATH=LSTM_PATH,
    encoder_path=ENCODER_PATH,
    return_accuracy=False,
    augmentar=False,
    n_aumentos=N_AUMENTOS,
):
    """
    Treina um modelo LSTM para reconhecimento de gestos dinâmicos (sequências temporais).
    Espera features no formato 3D: (amostras, frames, features_por_frame)

    augmentar : bool
        Se True, aplica data augmentation APENAS nos dados de treino (após o split),
        evitando data leakage.
    """
    X = np.array(features)
    y = np.array(labels)

    if X.ndim != 3:
        raise ValueError(
            f"LSTM requer dados 3D (amostras, frames, coordenadas). "
            f"Formato recebido: {X.shape}")

    num_amostras = X.shape[0]
    num_classes = len(set(labels))

    test_size = max(num_classes / num_amostras, 0.2)
    test_size = min(test_size, 0.3)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split sobre os rótulos originais (string) para poder augmentar o treino depois
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y_encoded)

    if augmentar:
        from landmark_augmentation import gerar_amostras_aumentadas

        X_aug, y_aug = gerar_amostras_aumentadas(
            X_train.tolist(),
            y_train_str.tolist(),
            mode="lstm",
            n_aumentos=n_aumentos)
        X_train = np.array(X_aug)
        y_train_str = np.array(y_aug)
        print(
            f"Treino após augmentation: {len(X_train)} amostras"
        )

    # Codificar labels após augmentation (o encoder já foi fitado no dataset completo)
    y_train = to_categorical(
        label_encoder.transform(y_train_str))
    y_test = to_categorical(
        label_encoder.transform(y_test_str))

    batch_size = min(8, len(X_train))
    print(
        f"LSTM usando batch_size={batch_size} (ajustado ao tamanho do dataset)"
    )

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(X_train.shape[1],
                         X_train.shape[2]),
            recurrent_dropout=0.1,
        ),
        Dropout(0.3),
        LSTM(128,
             return_sequences=False,
             recurrent_dropout=0.1),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss",
                               patience=LSTM_PATIENCE,
                               restore_best_weights=True,
                               verbose=1)

    classes_unicas = np.unique(y_train_str)
    pesos_array = compute_class_weight(
        "balanced", classes=classes_unicas, y=y_train_str)
    class_weight_dict = {
        int(label_encoder.transform([c])[0]): float(p)
        for c, p in zip(classes_unicas, pesos_array)
    }

    print("\nIniciando treinamento do LSTM...")
    model.fit(
        X_train,
        y_train,
        epochs=LSTM_EPOCHS,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test,
                                    y_test,
                                    verbose=0)
    print(
        f"\nAcurácia do modelo LSTM: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(LSTM_PATH), exist_ok=True)
    model.save(LSTM_PATH)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo LSTM salvo em {LSTM_PATH}")
    print(f"Label Encoder salvo em {encoder_path}")

    if return_accuracy:
        return accuracy

def train_knn(
    features,
    labels,
    KNN_PATH=KNN_PATH,
    n_clusters=N_CLUSTERS,
    return_accuracy=False,
    augmentar=False,
    n_aumentos=N_AUMENTOS,
):
    X = np.array(features)
    y = np.array(labels)

    if X.ndim != 3:
        raise ValueError(
            f"KNN requer dados 3D (amostras, frames, coordenadas). "
            f"Formato recebido: {X.shape}")

    sequence_length = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    if augmentar:
        from landmark_augmentation import gerar_amostras_aumentadas

        X_aug, y_aug = gerar_amostras_aumentadas(
            X_train.tolist(),
            y_train.tolist(),
            mode="lstm",
            n_aumentos=n_aumentos)
        X_train = np.array(X_aug)
        y_train = np.array(y_aug)
        print(
            f"Treino após augmentation: {len(X_train)} amostras"
        )

    # Aprende os centróides de "poses" sobre todos os frames de treino
    frames_treino = X_train.reshape(-1, X_train.shape[-1])
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=SEED,
                    n_init=10)
    kmeans.fit(frames_treino)

    def compactar(sequencias):
        histogramas = np.zeros((len(sequencias), n_clusters))
        for i, seq in enumerate(sequencias):
            clusters = kmeans.predict(seq)
            for c in clusters:
                histogramas[i, c] += 1
            histogramas[i] /= len(seq)
        return histogramas

    hist_train = compactar(X_train)
    hist_test = compactar(X_test)

    scaler = StandardScaler()
    hist_train = scaler.fit_transform(hist_train)
    hist_test = scaler.transform(hist_test)

    model = KNeighborsClassifier(weights="distance")

    print("\nIniciando treinamento do KNN...")
    model.fit(hist_train, y_train)

    y_pred = model.predict(hist_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do modelo KNN: {accuracy * 100:.2f}%")

    pacote = {
        "model": model,
        "scaler": scaler,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "sequence_length": sequence_length,
    }

    os.makedirs(os.path.dirname(KNN_PATH), exist_ok=True)
    with open(KNN_PATH, "wb") as f:
        pickle.dump(pacote, f)

    print(f"Modelo KNN salvo em {KNN_PATH}")

    if return_accuracy:
        return accuracy
