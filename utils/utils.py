import csv
import os
from datetime import datetime

import numpy as np

from utils.constants import (
    BEST_DIR,
    BEST_PATH,
    POSE_LANDMARKS_USADOS,
    POSE_OMBRO_DIREITO,
    POSE_OMBRO_ESQUERDO,
    RESULTADOS_TREINO_CSV,
    VIDEOS_POR_INTERPRETE,
    ZEROS_MAO,
)


def interprete_do_video(video_id):
    """
    Retorna o número do intérprete do vídeo a partir do video_id ({nome_gesto}_{id_video}).
    Usado como grupo nos splits, para um mesmo intérprete não estar no treino e na validação.
    """
    id_video = int(video_id.rsplit('_', 1)[-1])
    return (id_video - 1) // VIDEOS_POR_INTERPRETE + 1


def _landmarks_em_pixels(landmarks, largura, altura):
    """
    Converte landmarks do MediaPipe (x e y normalizados pela largura e altura da imagem,
    z na escala do x) para pixels, deixando os três eixos na mesma unidade e
    removendo a distorção causada pela proporção da imagem.
    """
    return np.array([[lm.x * largura, lm.y * altura, lm.z * largura] for lm in landmarks])


def normalizar_holistic(resultado, largura, altura):
    """
    Normaliza os landmarks do Holistic em relação ao corpo, retornando NUM_FEATURES features:
        [mão direita (63)] + [mão esquerda (63)] + [pose: ombros, cotovelos, pulsos e quadris (16, apenas x e y)]

    1. Converte todos os pontos para pixels (ver _landmarks_em_pixels).
    2. Origem de x e y no ponto médio entre os ombros → remove a posição da pessoa na imagem.
    3. Divide pela largura dos ombros em 2D → remove a distância da câmera, a resolução
       e o tamanho do intérprete, mantendo a posição das mãos em relação ao corpo.
    4. O z das mãos continua relativo ao próprio pulso (profundidade dos dedos), na mesma
       escala de x e y. O z do Pose não é usado: o MediaPipe o estima de forma exagerada
       e instável (com a pessoa parada, o z dos ombros oscila e os pulsos chegam a ~3 larguras
       de ombro à frente do corpo).

    Mão não detectada → bloco de zeros. Retorna None sem pose (não há referência do corpo).
    A mão direita é a mão direita da pessoa (right_hand_landmarks do Holistic).
    """
    if not resultado.pose_landmarks:
        return None

    pose = _landmarks_em_pixels(resultado.pose_landmarks, largura, altura)
    ombro_esq, ombro_dir = pose[POSE_OMBRO_ESQUERDO], pose[POSE_OMBRO_DIREITO]
    origem = (ombro_esq + ombro_dir) / 2
    origem[2] = 0.0
    escala = np.linalg.norm(ombro_esq[:2] - ombro_dir[:2])
    if escala < 1e-6:
        return None

    features = []
    for mao in (resultado.right_hand_landmarks, resultado.left_hand_landmarks):
        if not mao:
            features += ZEROS_MAO
            continue
        pontos = _landmarks_em_pixels(mao, largura, altura)
        pontos[:, 2] -= pontos[0, 2]
        features += ((pontos - origem) / escala).ravel().tolist()

    pose_xy = pose[list(POSE_LANDMARKS_USADOS), :2]
    features += ((pose_xy - origem[:2]) / escala).ravel().tolist()
    return features


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
    pulso = hand_landmarks[0] # Pulso
    base_ind = hand_landmarks[5] # Base Indicador
    base_min = hand_landmarks[17] # Base Mindinho

    escala_xy = ((base_ind.x - base_min.x)**2 + (base_ind.y - base_min.y)**2)**0.5

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
    feats_direita = ZEROS_MAO
    feats_esquerda = ZEROS_MAO

    for hand_lms, handed in zip(hand_landmarks_list, handedness_list):
        label = handed[0].category_name  # "Right" ou "Left"
        feats = normalizar_mao(hand_lms)
        if label == "Right":
            feats_direita = feats
        else:
            feats_esquerda = feats

    return feats_direita + feats_esquerda


def _salvar_log_treino(
    nome_modelo, acuracia, f1_score,
    n_amostras, augmentar, n_aumentos,
    parametros, tempo, epoca_parada
):
    """Persiste a acurácia de cada treino em logs/resultados_treino.csv."""
    log_path = RESULTADOS_TREINO_CSV
    cabecalho = [
        'data_hora', 'modelo', 'acuracia_%', 'f1_macro_%',
        'amostras_treino', 'augmentation', 'n_aumentos', 'tempo_min', 'epoca_parada', 'hiperparametros'
    ]
    novo_arquivo = not os.path.exists(log_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(cabecalho)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            nome_modelo,
            f'{acuracia * 100:.2f}',
            f'{f1_score * 100:.2f}',
            n_amostras,
            'sim' if augmentar else 'nao',
            n_aumentos if augmentar else 0,
            epoca_parada,
            f'{tempo:.4f}',
            parametros
        ])
    print(f"[LOG] Resultado salvo em '{log_path}'")


def ler_melhores_metricas(nome_modelo):
    """
    Lê as métricas do melhor treino do modelo em best/<nome_modelo>.txt.
    Retorna (f1_macro, acuracia) em %, ou None se o modelo ainda não tiver um melhor treino salvo.
    """
    best_path = BEST_PATH % nome_modelo
    if not os.path.exists(best_path):
        return None
    metricas = {}
    with open(best_path, encoding='utf-8') as f:
        for linha in f:
            if ':' in linha:
                chave, valor = linha.split(':', 1)
                metricas[chave.strip()] = float(valor)
    return metricas['f1-macro'], metricas['acuracia']


def salvar_melhores_metricas(nome_modelo, f1_macro, acuracia):
    """Grava as métricas (em %) do novo melhor treino do modelo em best/<nome_modelo>.txt."""
    os.makedirs(BEST_DIR, exist_ok=True)
    best_path = BEST_PATH % nome_modelo
    with open(best_path, 'w', encoding='utf-8') as f:
        f.write(f"f1-macro:{f1_macro:.2f}\n")
        f.write(f"acuracia:{acuracia:.2f}")
    print(f"[BEST] Métricas salvas em '{best_path}'")


def eh_melhor_treino(nome_modelo, f1_macro, acuracia):
    """
    Compara as métricas (em %) do treino atual com as do melhor treino salvo do modelo.
    O treino atual é melhor se tiver F1-Macro maior ou, em caso de empate, acurácia maior.
    Sem melhor treino salvo, o treino atual é sempre o melhor.
    """
    melhor = ler_melhores_metricas(nome_modelo)
    if melhor is None:
        print(f"[BEST] Nenhum treino anterior de {nome_modelo} em '{BEST_DIR}/'")
        return True
    f1_melhor, acc_melhor = melhor
    atual = (round(f1_macro, 2), round(acuracia, 2))
    print(f"[BEST] {nome_modelo} | atual: F1-Macro {atual[0]:.2f}% e Acurácia {atual[1]:.2f}% "
          f"| melhor: F1-Macro {f1_melhor:.2f}% e Acurácia {acc_melhor:.2f}%")
    return atual > (f1_melhor, acc_melhor)
