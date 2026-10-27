"""
landmark_augmentation.py
========================
Data augmentation diretamente nos landmarks extraídos pelo MediaPipe.

Suporta os dois formatos de features do projeto:
  - 126 features (extração antiga, HandLandmarker): [mão direita (0-62), mão esquerda (63-125)],
    cada mão relativa ao próprio pulso.
  - 142 features (extração com o Holistic): [mão direita (0-62), mão esquerda (63-125), pose (126-141)],
    todos os pontos relativos ao corpo (origem no centro dos ombros); a pose tem apenas x e y.
Mão ausente → bloco de zeros.

Transformações:
  1. Ruído gaussiano  — simula tremor / imprecisão do sensor
  2. Escala           — simula distância diferente da câmera
  3. Rotação 2D       — simula ângulo de filmagem diferente (plano XY)
  4. Espelhamento     — inverte X; também troca direita↔esquerda (mãos e pares do Pose)

Para sequências LSTM os parâmetros aleatórios são pré-gerados uma vez e
aplicados a todos os frames, garantindo coerência temporal.
"""

import numpy as np

from utils.constants import (
    AUG_NOISE_INTENSITY,
    AUG_ROTATION_MAX_DEGREES,
    AUG_SCALE_MAX,
    AUG_SCALE_MIN,
    COORDINATES_PER_LANDMARK,
    COORDINATES_PER_POSE_LANDMARK,
    N_AUMENTOS,
    NUM_FEATURES,
    NUM_FEATURES_MAOS,
    NUM_LANDMARKS_PER_HAND,
    POSE_LANDMARKS_USADOS,
    POSE_PARES_ESPELHO,
    SEED,
)

# Pontos de cada bloco do frame, com o frame no formato (n_pontos, 3)
# (a pose, que tem apenas x e y, recebe um z nulo durante as transformações)
MAO_D = slice(0, NUM_LANDMARKS_PER_HAND)
MAO_E = slice(NUM_LANDMARKS_PER_HAND, 2 * NUM_LANDMARKS_PER_HAND)
POSE = slice(2 * NUM_LANDMARKS_PER_HAND, 2 * NUM_LANDMARKS_PER_HAND + len(POSE_LANDMARKS_USADOS))


def _ordem_pose_espelhada():
    """Ordem dos pontos do Pose após o espelhamento: cada ponto troca de lugar com o seu par."""
    par = {}
    for a, b in POSE_PARES_ESPELHO:
        par[a], par[b] = b, a
    posicao = {lm: i for i, lm in enumerate(POSE_LANDMARKS_USADOS)}
    return [posicao[par.get(lm, lm)] for lm in POSE_LANDMARKS_USADOS]


ORDEM_POSE_ESPELHADA = _ordem_pose_espelhada()


# Helpers das funções principais
def _eh_ausente(pts):
    """True se o bloco (mão ou pose) está ausente (bloco de zeros)."""
    return np.all(np.abs(pts) < 1e-9)


# Transformações

def _ruido(pts, intensidade=AUG_NOISE_INTENSITY):
    return pts + np.random.normal(0, intensidade, pts.shape).astype(np.float32)


def _escala(pts, fator_min=AUG_SCALE_MIN, fator_max=AUG_SCALE_MAX):
    return pts * np.random.uniform(fator_min, fator_max)


def _rotacao_2d(pts, angulo_max_graus=AUG_ROTATION_MAX_DEGREES):
    angulo = np.radians(
        np.random.uniform(-angulo_max_graus, angulo_max_graus)
    )
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = cos_a * x - sin_a * y
    pts[:, 1] = sin_a * x + cos_a * y
    return pts


def _espelhamento(pts):
    """Inverte o eixo X. Deve ser combinado com troca de lados."""
    pts[:, 0] = -pts[:, 0]
    return pts



# Wrapper para aplicar as transformações
def _aplicar(
    pts, t, 
    ruido_frame=None,
    escala_fator=None,
    cos_a=None,
    sin_a=None
):
    """
    Aplica transformação t a um bloco de pontos (n, 3), em torno da origem
    (pulso no formato de 126 features, centro dos ombros no formato do Holistic).
    Se os parâmetros pré-fixados forem fornecidos, usa-os (modo sequência LSTM).
    Caso contrário, sorteia na hora (modo frame isolado).
    """
    if t == _ruido:
        return pts + (
            ruido_frame if ruido_frame is not None
            else np.random.normal(
                0, AUG_NOISE_INTENSITY, pts.shape
            ).astype(np.float32))
    elif t == _escala:
        return pts * (
            escala_fator if escala_fator is not None
            else np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
        )
    elif t == _rotacao_2d:
        if cos_a is None:
            ang = np.radians(
                np.random.uniform(-AUG_ROTATION_MAX_DEGREES, AUG_ROTATION_MAX_DEGREES)
            )
            cos_a, sin_a = np.cos(ang), np.sin(ang)

        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        nova = pts.copy()
        nova[:, 0] = cos_a * x - sin_a * y
        nova[:, 1] = sin_a * x + cos_a * y
        return nova
    
    elif t == _espelhamento:
        # Espelhamento chamado aqui só em contexto de bloco único
        return _espelhamento(pts.copy())
    else:
        return t(pts)


# ---------------------------------------------------------------------------
# Núcleo: processa um frame com params opcionais
# ---------------------------------------------------------------------------


def _processar_frame(
    frame_arr,
    transformacoes,
    ruido=None,
    escala_fator=None,
    cos_a=None,
    sin_a=None,
):
    """
    Aplica transformações a um frame (126 ou 142 features).

    Espelhamento:
      1. Inverte X de todos os pontos
      2. Troca direita↔esquerda — porque a imagem espelhada tem as mãos invertidas
         (no Pose, cada ponto também troca de lugar com o seu par do outro lado do corpo)

    Para blocos ausentes (bloco de zeros) nenhuma transformação é aplicada,
    preservando os zeros que indicam "mão não detectada".
    No formato de 126 features as mãos são recentralizadas no pulso ao final.
    """
    tem_pose = len(frame_arr) == NUM_FEATURES
    if not tem_pose and len(frame_arr) != NUM_FEATURES_MAOS:
        raise ValueError(
            f"Frame com {len(frame_arr)} features; esperado {NUM_FEATURES_MAOS} ou {NUM_FEATURES}"
        )

    frame_arr = np.asarray(frame_arr, dtype=np.float32)
    pts = frame_arr[:NUM_FEATURES_MAOS].reshape(-1, COORDINATES_PER_LANDMARK)
    if tem_pose:
        pose = frame_arr[NUM_FEATURES_MAOS:].reshape(-1, COORDINATES_PER_POSE_LANDMARK)
        pose = np.pad(pose, ((0, 0), (0, COORDINATES_PER_LANDMARK - COORDINATES_PER_POSE_LANDMARK)))
        pts = np.concatenate([pts, pose])
    blocos = [MAO_D, MAO_E] + ([POSE] if tem_pose else [])
    presentes = [b for b in blocos if not _eh_ausente(pts[b])]

    # Separar espelhamento das demais (requer lógica de troca)
    espelhar = _espelhamento in transformacoes
    outras = [t for t in transformacoes if t != _espelhamento]

    for t in outras:
        for b in presentes:
            pts[b] = _aplicar(
                pts[b], t, ruido[b] if ruido is not None else None,
                escala_fator, cos_a, sin_a
            )

    if espelhar:
        for b in presentes:
            pts[b] = _espelhamento(pts[b].copy())
        # troca: o que era direita vira esquerda
        pts[MAO_D], pts[MAO_E] = pts[MAO_E].copy(), pts[MAO_D].copy()
        if tem_pose:
            pts[POSE] = pts[POSE][ORDEM_POSE_ESPELHADA]

    if not tem_pose:
        for b in (MAO_D, MAO_E):
            if not _eh_ausente(pts[b]):
                pts[b] = pts[b] - pts[b][0]
        return pts.flatten().tolist()

    return pts[:POSE.start].flatten().tolist() + pts[POSE, :COORDINATES_PER_POSE_LANDMARK].flatten().tolist()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def augmentar_sequencia(sequencia, transformacoes):
    """
    Aplica as mesmas transformações em TODOS os frames da sequência LSTM.

    Parâmetros aleatórios pré-gerados uma vez para toda a sequência,
    garantindo coerência temporal: o ângulo de rotação é o mesmo nos
    20 frames, evitando criar movimentos artificiais.
    """
    n = len(sequencia)

    escala_fator = np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
    ang = np.radians(
        np.random.uniform(-AUG_ROTATION_MAX_DEGREES, AUG_ROTATION_MAX_DEGREES)
    )
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    n_pontos = MAO_E.stop + (POSE.stop - POSE.start if len(sequencia[0]) == NUM_FEATURES else 0)
    ruido = np.random.normal(
        0, AUG_NOISE_INTENSITY,
        (n, n_pontos, COORDINATES_PER_LANDMARK)
    ).astype(np.float32)

    resultado = []
    for fi, frame in enumerate(sequencia):
        arr = np.array(frame, dtype=np.float32)
        r = _processar_frame(
            arr,
            transformacoes,
            ruido=ruido[fi],
            escala_fator=escala_fator,
            cos_a=cos_a,
            sin_a=sin_a,
        )
        resultado.append(r)

    return resultado


def gerar_amostras_aumentadas(features, labels, grupos, n_aumentos=N_AUMENTOS, seed=SEED):
    """
    Gera amostras augmentadas para o dataset inteiro.
    Suporta 126 (apenas mãos) e 142 (Holistic) features automaticamente.
    n_aumentos : augmentações por amostra original.
        Com n_aumentos=5: 1 espelhamento + 4 combinações = +5 → total 6×.
    """
    np.random.seed(seed)

    transformacoes_disponiveis = [_ruido, _escala, _rotacao_2d]

    features_aug = list(features)
    labels_aug = list(labels)
    grupo_aug = list(grupos)

    originais_por_classe = {}
    for l in labels:
        originais_por_classe[l] = originais_por_classe.get(l, 0) + 1

    print("\n--- Augmentation de Landmarks ---")
    print(f"Amostras originais: {len(features)}")
    print(f"Gerando {n_aumentos} variações por amostra...")

    for amostra, label, grupo in zip(features, labels, grupos):

        # Sempre gera uma amostra espelhada
        esp = augmentar_sequencia(amostra, [_espelhamento])
        features_aug.append(esp)
        labels_aug.append(label)
        grupo_aug.append(grupo)

        # 2. Combinações aleatórias das demais transformações
        for _ in range(n_aumentos - 1):
            n_t = np.random.randint(1, len(transformacoes_disponiveis) + 1)
            escolhidas = np.random.choice(
                transformacoes_disponiveis,
                size=n_t,
                replace=False
            ).tolist()

            nova = augmentar_sequencia(amostra, escolhidas)
            features_aug.append(nova)
            labels_aug.append(label)
            grupo_aug.append(grupo)

    print(f"Amostras após augmentation: {len(features_aug)}")
    print(f"Aumento: {len(features_aug) / max(len(features), 1):.1f}×")

    dist = {}
    for l in labels_aug:
        dist[l] = dist.get(l, 0) + 1

    print("\nDistribuição por classe:")
    for classe, qtd in sorted(dist.items()):
        orig = originais_por_classe.get(classe, 0)
        print(f"  {classe}: {orig} → {qtd} amostras")

    return np.array(features_aug).astype(np.float32), np.array(labels_aug), np.array(grupo_aug)


if __name__ == "__main__":
    print("\n=== Teste: seq T=20, 126 features ===")
    f_lstm2 = [np.random.randn(20, 126).tolist() for _ in range(4)]
    l_lstm = ["oi"] * 2 + ["tchau"] * 2
    grupos = [1] * 4
    print(np.array(f_lstm2).shape)
    f_aug, l_aug, grupos_aug = gerar_amostras_aumentadas(f_lstm2, l_lstm, grupos=grupos, n_aumentos=10)
    assert len(f_aug[0]) == 20 and len(f_aug[0][0]) == 126
    print("OK")