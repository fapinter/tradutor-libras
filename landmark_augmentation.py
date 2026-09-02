"""
landmark_augmentation.py
========================
Data augmentation diretamente nos landmarks extraídos pelo MediaPipe.

Suporta dois modos de feature:
  - 63  features: 1 mão  (21 landmarks × 3 coords)
  - 126 features: 2 mãos (21 landmarks × 3 coords × 2 mãos)
    Ordem: [mão direita (0-62), mão esquerda (63-125)]
    Mão ausente → bloco de zeros.

Transformações:
  1. Ruído gaussiano  — simula tremor / imprecisão do sensor
  2. Escala           — simula distância diferente da câmera
  3. Rotação 2D       — simula ângulo de filmagem diferente (plano XY)
  4. Espelhamento     — inverte X; para 2 mãos, também troca direita↔esquerda

Para sequências LSTM os parâmetros aleatórios são pré-gerados uma vez e
aplicados a todos os frames, garantindo coerência temporal.
"""

import numpy as np

from utils.constants import (
    AUG_NOISE_INTENSITY,
    AUG_ROTATION_MAX_DEGREES,
    AUG_SCALE_MAX,
    AUG_SCALE_MIN,
    N_AUMENTOS,
    N_MAO,
    NUM_FEATURES,
    SEED,
)

# ---------------------------------------------------------------------------
# Helpers de reshape
# ---------------------------------------------------------------------------


def _reshape(frame_mao):
    """Lista plana (63,) → matriz (21, 3)."""
    return np.array(frame_mao,
                    dtype=np.float32).reshape(21, 3)


def _flatten(pts):
    """Matriz (21, 3) → lista plana (63,)."""
    return pts.flatten().tolist()


def _eh_ausente(pts):
    """True se a mão está ausente (bloco de zeros)."""
    return np.all(np.abs(pts) < 1e-9)


# ---------------------------------------------------------------------------
# Transformações individuais — operam em (21, 3) de uma mão
# ---------------------------------------------------------------------------


def _ruido(pts, intensidade=AUG_NOISE_INTENSITY):
    return pts + np.random.normal(
        0, intensidade, pts.shape).astype(np.float32)


def _escala(pts,
            fator_min=AUG_SCALE_MIN,
            fator_max=AUG_SCALE_MAX):
    return pts * np.random.uniform(fator_min, fator_max)


def _rotacao_2d(pts,
                angulo_max_graus=AUG_ROTATION_MAX_DEGREES):
    angulo = np.radians(
        np.random.uniform(-angulo_max_graus,
                          angulo_max_graus))
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = cos_a * x - sin_a * y
    pts[:, 1] = sin_a * x + cos_a * y
    return pts


def _espelhamento(pts):
    """Inverte o eixo X. Para 2 mãos, deve ser combinado com troca de lados."""
    pts[:, 0] = -pts[:, 0]
    return pts


# ---------------------------------------------------------------------------
# Helper interno: aplica uma transformação com parâmetros pré-fixados
# ---------------------------------------------------------------------------


def _aplicar(pts,
             t,
             ruido_frame=None,
             escala_fator=None,
             cos_a=None,
             sin_a=None):
    """
    Aplica transformação t a uma mão (21, 3).
    Se os parâmetros pré-fixados forem fornecidos, usa-os (modo sequência LSTM).
    Caso contrário, sorteia na hora (modo frame isolado).
    """
    if t == _ruido:
        return pts + (ruido_frame if ruido_frame is not None
                      else np.random.normal(
                          0, AUG_NOISE_INTENSITY,
                          pts.shape).astype(np.float32))
    elif t == _escala:
        return pts * (escala_fator if escala_fator
                      is not None else np.random.uniform(
                          AUG_SCALE_MIN, AUG_SCALE_MAX))
    elif t == _rotacao_2d:
        if cos_a is None:
            ang = np.radians(
                np.random.uniform(-AUG_ROTATION_MAX_DEGREES,
                                  AUG_ROTATION_MAX_DEGREES))
            cos_a, sin_a = np.cos(ang), np.sin(ang)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        nova = pts.copy()
        nova[:, 0] = cos_a * x - sin_a * y
        nova[:, 1] = sin_a * x + cos_a * y
        return nova
    elif t == _espelhamento:
        # Espelhamento chamado aqui só em contexto de mão única
        return _espelhamento(pts.copy())
    else:
        return t(pts)


# ---------------------------------------------------------------------------
# Núcleo: processa um frame (63 ou 126 features) com params opcionais
# ---------------------------------------------------------------------------


def _processar_frame(
    frame_arr,
    transformacoes,
    ruido_a=None,
    ruido_b=None,
    escala_fator=None,
    cos_a=None,
    sin_a=None,
):
    """
    Aplica transformações a um frame, detectando automaticamente se há 1 ou 2 mãos.

    Para 2 mãos, o espelhamento:
      1. Inverte X de ambas as mãos
      2. Troca direita↔esquerda — porque a imagem espelhada tem as mãos invertidas

    Para mãos ausentes (bloco de zeros) nenhuma transformação é aplicada,
    preservando os zeros que indicam "mão não detectada".
    """
    duas_maos = len(frame_arr) == NUM_FEATURES

    if duas_maos:
        mao_d = _reshape(frame_arr[:N_MAO])
        mao_e = _reshape(frame_arr[N_MAO:])

        tem_d = not _eh_ausente(mao_d)
        tem_e = not _eh_ausente(mao_e)

        # Separar espelhamento das demais (requer lógica de troca)
        espelhar = _espelhamento in transformacoes
        outras = [
            t for t in transformacoes if t != _espelhamento
        ]

        for t in outras:
            if tem_d:
                mao_d = _aplicar(mao_d, t, ruido_a,
                                 escala_fator, cos_a, sin_a)
            if tem_e:
                mao_e = _aplicar(mao_e, t, ruido_b,
                                 escala_fator, cos_a, sin_a)

        if espelhar:
            nd = _espelhamento(
                mao_d.copy()) if tem_d else mao_d
            ne = _espelhamento(
                mao_e.copy()) if tem_e else mao_e
            mao_d, mao_e = ne, nd  # troca: o que era direita vira esquerda

        if tem_d:
            mao_d = mao_d - mao_d[0]
        if tem_e:
            mao_e = mao_e - mao_e[0]

        return _flatten(mao_d) + _flatten(mao_e)

    else:  # mão única (63 features)
        pts = _reshape(frame_arr)
        for t in transformacoes:
            pts = _aplicar(pts, t, ruido_a, escala_fator,
                           cos_a, sin_a)
        pts = pts - pts[0]
        return _flatten(pts)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def augmentar_frame(frame, transformacoes):
    """
    Aplica transformações a um frame.
    Aceita 63 features (1 mão) ou 126 features (2 mãos).
    """
    return _processar_frame(
        np.array(frame, dtype=np.float32), transformacoes)


def augmentar_sequencia(sequencia, transformacoes):
    """
    Aplica as mesmas transformações em TODOS os frames da sequência LSTM.
    Aceita 63 features (1 mão) ou 126 features (2 mãos).

    Parâmetros aleatórios pré-gerados uma vez para toda a sequência,
    garantindo coerência temporal: o ângulo de rotação é o mesmo nos
    20 frames, evitando criar movimentos artificiais.
    """
    n = len(sequencia)
    frame_len = len(sequencia[0]) if sequencia else N_MAO

    escala_fator = np.random.uniform(AUG_SCALE_MIN,
                                     AUG_SCALE_MAX)
    ang = np.radians(
        np.random.uniform(-AUG_ROTATION_MAX_DEGREES,
                          AUG_ROTATION_MAX_DEGREES))
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    if frame_len == NUM_FEATURES:
        ruido_d = np.random.normal(0, AUG_NOISE_INTENSITY,
                                   (n, 21, 3)).astype(
                                       np.float32)
        ruido_e = np.random.normal(0, AUG_NOISE_INTENSITY,
                                   (n, 21, 3)).astype(
                                       np.float32)
    else:
        ruido_u = np.random.normal(0, AUG_NOISE_INTENSITY,
                                   (n, 21, 3)).astype(
                                       np.float32)

    resultado = []
    for fi, frame in enumerate(sequencia):
        arr = np.array(frame, dtype=np.float32)
        if frame_len == NUM_FEATURES:
            r = _processar_frame(
                arr,
                transformacoes,
                ruido_a=ruido_d[fi],
                ruido_b=ruido_e[fi],
                escala_fator=escala_fator,
                cos_a=cos_a,
                sin_a=sin_a,
            )
        else:
            r = _processar_frame(
                arr,
                transformacoes,
                ruido_a=ruido_u[fi],
                escala_fator=escala_fator,
                cos_a=cos_a,
                sin_a=sin_a,
            )
        resultado.append(r)

    return resultado


def gerar_amostras_aumentadas(features,
                              labels,
                              mode="rf",
                              n_aumentos=N_AUMENTOS,
                              seed=SEED):
    """
    Gera amostras augmentadas para o dataset inteiro.
    Suporta 63 e 126 features automaticamente.

    mode : "rf"   → features 2D (N, F)
           "lstm" → features 3D (N, T, F)
    n_aumentos : augmentações por amostra original.
        Com n_aumentos=5: 1 espelhamento + 4 combinações = +5 → total 6×.
    """
    np.random.seed(seed)

    transformacoes_disponiveis = [
        _ruido, _escala, _rotacao_2d
    ]

    features_aug = list(features)
    labels_aug = list(labels)

    originais_por_classe = {}
    for l in labels:
        originais_por_classe[l] = originais_por_classe.get(
            l, 0) + 1

    print("\n--- Augmentation de Landmarks ---")
    print(f"Amostras originais: {len(features)}")
    print(f"Gerando {n_aumentos} variações por amostra...")

    for amostra, label in zip(features, labels):
        # 1. Espelhamento (sempre — é a mais valiosa, simula mão não-dominante)
        if mode == "rf":
            esp = _processar_frame(
                np.array(amostra, dtype=np.float32),
                [_espelhamento])
            features_aug.append(esp)
        else:
            esp = augmentar_sequencia(amostra,
                                      [_espelhamento])
            features_aug.append(esp)
        labels_aug.append(label)

        # 2. Combinações aleatórias das demais transformações
        for _ in range(n_aumentos - 1):
            n_t = np.random.randint(
                1,
                len(transformacoes_disponiveis) + 1)
            escolhidas = np.random.choice(
                transformacoes_disponiveis,
                size=n_t,
                replace=False).tolist()

            if mode == "rf":
                nova = augmentar_frame(amostra, escolhidas)
            else:
                nova = augmentar_sequencia(
                    amostra, escolhidas)
            features_aug.append(nova)
            labels_aug.append(label)

    print(
        f"Amostras após augmentation: {len(features_aug)}")
    print(
        f"Aumento: {len(features_aug) / max(len(features), 1):.1f}×"
    )

    dist = {}
    for l in labels_aug:
        dist[l] = dist.get(l, 0) + 1
    print("\nDistribuição por classe:")
    for classe, qtd in sorted(dist.items()):
        orig = originais_por_classe.get(classe, 0)
        print(f"  {classe}: {orig} → {qtd} amostras")

    return features_aug, labels_aug


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste: 1 mão (63 features) ===")
    np.random.seed(0)
    f_rf = [np.random.randn(63).tolist() for _ in range(6)]
    l_rf = ["oi"] * 3 + ["tchau"] * 3
    f_aug, l_aug = gerar_amostras_aumentadas(f_rf,
                                             l_rf,
                                             mode="rf",
                                             n_aumentos=5)
    assert len(f_aug) == len(l_aug) and len(f_aug[0]) == 63
    print("[RF 1 mão] OK")

    print("\n=== Teste: 2 mãos (126 features) ===")
    f_rf2 = [
        np.random.randn(126).tolist() for _ in range(6)
    ]
    f_aug2, l_aug2 = gerar_amostras_aumentadas(f_rf2,
                                               l_rf,
                                               mode="rf",
                                               n_aumentos=5)
    assert len(f_aug2[0]) == 126
    print("[RF 2 mãos] OK")

    print(
        "\n=== Teste: LSTM 1 mão (seq T=20, 63 features) ==="
    )
    f_lstm = [
        np.random.randn(20, 63).tolist() for _ in range(4)
    ]
    l_lstm = ["oi"] * 2 + ["tchau"] * 2
    f_aug3, l_aug3 = gerar_amostras_aumentadas(f_lstm,
                                               l_lstm,
                                               mode="lstm",
                                               n_aumentos=5)
    assert len(f_aug3[0]) == 20 and len(f_aug3[0][0]) == 63
    print("[LSTM 1 mão] OK")

    print(
        "\n=== Teste: LSTM 2 mãos (seq T=20, 126 features) ==="
    )
    f_lstm2 = [
        np.random.randn(20, 126).tolist() for _ in range(4)
    ]
    f_aug4, l_aug4 = gerar_amostras_aumentadas(f_lstm2,
                                               l_lstm,
                                               mode="lstm",
                                               n_aumentos=5)
    assert len(f_aug4[0]) == 20 and len(f_aug4[0][0]) == 126
    print("[LSTM 2 mãos] OK")

    print("\nTodos os testes passaram!")
