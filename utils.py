import numpy as np
from sklearn.cluster import KMeans

NUM_FEATURES = 126  # 21 landmarks × 3 coords × 2 mãos
_N_MAO = 63               # features de uma mão
_ZEROS_MAO = [0.0] * _N_MAO  # placeholder para mão ausente


def normalizar_mao(hand_landmarks):
    """
    Normaliza os landmarks de UMA mão (63 features).

    1. Subtrai o pulso (lm 0) em X, Y e Z → remove posição absoluta.
    2. Divide X e Y pela largura da palma em 2D (distância XY entre a base do
       indicador lm 5 e a base do mínimo lm 17) → remove variação de escala/câmera
       sem tocar no eixo Z.
    3. Z permanece apenas centrado no pulso → preserva profundidade relativa
       entre os dedos.
    """
    pulso = hand_landmarks[0]
    base_ind = hand_landmarks[5]
    base_min = hand_landmarks[17]

    escala_xy = (
        (base_ind.x - base_min.x) ** 2 +
        (base_ind.y - base_min.y) ** 2
    ) ** 0.5

    if escala_xy < 1e-6:
        escala_xy = 1.0

    landmarks = []
    for lm in hand_landmarks:
        landmarks.append((lm.x - pulso.x) / escala_xy)
        landmarks.append((lm.y - pulso.y) / escala_xy)
        landmarks.append(lm.z - pulso.z)
    return landmarks


def extrair_ambas_maos(hand_landmarks_list, handedness_list):
    """
    Extrai e concatena as features das duas mãos em ordem consistente:
        [mão direita (63 features)] + [mão esquerda (63 features)] = 126 features

    Mão não detectada → bloco de zeros (o modelo aprende que zeros = ausente).
    A ordem direita/esquerda é determinada pela label de lateralidade do MediaPipe.
    """
    feats_direita = _ZEROS_MAO
    feats_esquerda = _ZEROS_MAO

    for hand_lms, handed in zip(hand_landmarks_list, handedness_list):
        label = handed[0].category_name  # "Right" ou "Left"
        feats = normalizar_mao(hand_lms)
        if label == "Right":
            feats_direita = feats
        else:
            feats_esquerda = feats

    return feats_direita + feats_esquerda


def kmeans_temporal_single(seq_array, n_clusters):
    """
    Aplica K-Means temporal a uma única sequência de frames.
    Implementação da compactação temporal descrita em Caiafa et al. (SBrT 2023).
    """
    seq_array = np.array(seq_array, dtype=float)
    n_frames = len(seq_array)
    m = min(n_clusters, n_frames)

    km = KMeans(n_clusters=m, random_state=42, n_init=10)
    km.fit(seq_array)
    assignments = km.labels_

    centroid_order = []
    for c in range(m):
        frame_idxs = np.where(assignments == c)[0]
        median_idx = float(np.median(frame_idxs)) if len(frame_idxs) > 0 else float(c)
        centroid_order.append((median_idx, c))
    centroid_order.sort(key=lambda x: x[0])

    sorted_centroids = km.cluster_centers_[[c for _, c in centroid_order]]

    if m < n_clusters:
        pad = np.zeros((n_clusters - m, seq_array.shape[1]))
        sorted_centroids = np.vstack([sorted_centroids, pad])

    return sorted_centroids.flatten()


def aplicar_kmeans_temporal(sequences, n_clusters):
    """
    Aplica o K-Means temporal em um conjunto de sequências (matriz ou lista 3D).
    """
    result = []
    for seq in sequences:
        result.append(kmeans_temporal_single(seq, n_clusters))
    return np.array(result)
