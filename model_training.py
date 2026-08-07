import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def train_random_forest(
    features,
    labels,
    model_path="models/sign_model.pkl",
    return_accuracy=False,
    augmentar=False,
    n_aumentos=5,
):
    """
    Treina um modelo Random Forest para gestos estáticos (frame-a-frame).
    Espera features no formato 2D: (amostras, features_por_frame)

    augmentar : bool
        Se True, aplica data augmentation APENAS nos dados de treino (após o split),
        evitando data leakage.
    """
    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if augmentar:
        from landmark_augmentation import gerar_amostras_aumentadas

        X_aug, y_aug = gerar_amostras_aumentadas(
            X_train.tolist(), y_train.tolist(), mode="rf", n_aumentos=n_aumentos
        )
        X_train = np.array(X_aug)
        y_train = np.array(y_aug)
        print(f"Treino após augmentation: {len(X_train)} amostras")

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do Random Forest: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo Random Forest salvo em {model_path}")

    if return_accuracy:
        return accuracy


from utils import aplicar_kmeans_temporal as _aplicar_kmeans_temporal


def train_knn(
    features,
    labels,
    model_path="models/knn_sign_model.pkl",
    return_accuracy=False,
    n_clusters=10,
    augmentar=False,
    n_aumentos=5,
):
    """
    Treina KNN com pré-processamento K-Means temporal (Caiafa et al., SBrT 2023).

    augmentar : bool
        Se True, aplica data augmentation APENAS nos dados de treino (após o split).
    """
    features_array = np.array(features)

    if features_array.ndim == 3:
        usa_kmeans = True
        seq_len = features_array.shape[1]
    else:
        usa_kmeans = False
        seq_len = None

    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels, test_size=0.2, random_state=42, stratify=labels
    )

    if augmentar:
        from landmark_augmentation import gerar_amostras_aumentadas

        mode = "lstm" if usa_kmeans else "rf"
        X_aug, y_aug = gerar_amostras_aumentadas(
            X_train.tolist(), list(y_train), mode=mode, n_aumentos=n_aumentos
        )
        X_train = np.array(X_aug)
        y_train = np.array(y_aug)
        print(f"Treino após augmentation: {len(X_train)} amostras")

    if usa_kmeans:
        print(
            f"K-Means temporal: {n_clusters} centróides por janela de {seq_len} frames..."
        )
        X_train_2d = _aplicar_kmeans_temporal(X_train, n_clusters)
        X_test_2d = _aplicar_kmeans_temporal(X_test, n_clusters)
    else:
        print("Modo direto (frame-a-frame, sem K-Means temporal)...")
        X_train_2d = X_train
        X_test_2d = X_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    num_classes = len(set(list(y_train) + list(y_test)))
    k = max(1, min(5, len(X_train_scaled) // num_classes))
    print(f"KNN usando k={k} (ajustado ao tamanho do dataset)")

    model = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Acurácia do KNN: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler": scaler,
                "n_clusters": n_clusters,
                "usa_kmeans": usa_kmeans,
                "sequence_length": seq_len,
            },
            f,
        )
    print(f"Modelo KNN salvo em {model_path}")

    if return_accuracy:
        return accuracy


def train_lstm(
    features,
    labels,
    model_path="models/lstm_sign_model.h5",
    encoder_path="models/label_encoder.pkl",
    return_accuracy=False,
    augmentar=False,
    n_aumentos=5,
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
            f"Formato recebido: {X.shape}"
        )

    num_amostras = X.shape[0]
    num_classes = len(set(labels))

    test_size = max(num_classes / num_amostras, 0.2)
    test_size = min(test_size, 0.3)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split sobre os rótulos originais (string) para poder augmentar o treino depois
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y_encoded
    )

    if augmentar:
        from landmark_augmentation import gerar_amostras_aumentadas

        X_aug, y_aug = gerar_amostras_aumentadas(
            X_train.tolist(), y_train_str.tolist(), mode="lstm", n_aumentos=n_aumentos
        )
        X_train = np.array(X_aug)
        y_train_str = np.array(y_aug)
        print(f"Treino após augmentation: {len(X_train)} amostras")

    # Codificar labels após augmentation (o encoder já foi fitado no dataset completo)
    y_train = to_categorical(label_encoder.transform(y_train_str))
    y_test = to_categorical(label_encoder.transform(y_test_str))

    batch_size = min(8, len(X_train))
    print(f"LSTM usando batch_size={batch_size} (ajustado ao tamanho do dataset)")

    model = Sequential(
        [
            LSTM(
                64,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                recurrent_dropout=0.1,
            ),
            Dropout(0.3),
            LSTM(128, return_sequences=False, recurrent_dropout=0.1),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    # Pesos por classe: compensa desequilíbrio sem precisar de mais dados.
    # Sem isso, classes com mais sequências (ex: acabar=72) dominam o gradiente
    # e o modelo aprende a ignorar classes pequenas (ex: agora=12, cego=12).
    classes_unicas = np.unique(y_train_str)
    pesos_array = compute_class_weight(
        "balanced", classes=classes_unicas, y=y_train_str
    )
    # Keras espera dict {índice_int: peso}, onde índice = posição no label_encoder
    class_weight_dict = {
        int(label_encoder.transform([c])[0]): float(p)
        for c, p in zip(classes_unicas, pesos_array)
    }

    print("\nIniciando treinamento do LSTM...")
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia do modelo LSTM: {accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Modelo LSTM salvo em {model_path}")
    print(f"Label Encoder salvo em {encoder_path}")

    if return_accuracy:
        return accuracy
