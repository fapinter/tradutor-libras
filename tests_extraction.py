import os
from collections import namedtuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from feature_extraction import preprocessImage
from utils.constants import (
    DEFAULT_STEP,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    LANDMARKER_PATH,
    LOGS_DIR,
    NUM_FEATURES_MAOS,
    NUM_HANDS,
    SEQUENCE_LENGTH,
    ZEROS_FRAME_MAOS,
    ZEROS_MAO,
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import normalizar_mao

# Ponto no mesmo formato dos landmarks do MediaPipe, para reutilizar normalizar_mao
Ponto = namedtuple("Ponto", ["x", "y", "z"])


def nome_configuracao(preprocess=False):
    """Sufixo usado nos nomes do CSV e do log, para cada configuração não sobrescrever as outras."""
    nome = "mediapipe"
    if preprocess:
        nome += "_preprocess"
    return nome


def detectar_mediapipe(landmarker, img_rgb):
    """
    Retorna as mãos detectadas pelo MediaPipe no formato {"Right"|"Left": array (21, 3)},
    com coordenadas normalizadas pela imagem (antes da normalização pelo pulso).
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)

    maos = {}
    for hand_lms, handed in zip(results.hand_landmarks, results.handedness):
        label = handed[0].category_name  # "Right" ou "Left"
        maos[label] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
    return maos


def maos_para_features(maos):
    """Concatena [mão direita (63)] + [mão esquerda (63)], com zeros para mão ausente."""
    feats = []
    for label in ("Right", "Left"):
        mao = maos.get(label)
        if mao is None:
            feats += ZEROS_MAO
        else:
            feats += normalizar_mao([Ponto(*lm) for lm in mao])
    return feats


def extract_features_from_directory(
    dataset_root_dir,
    output_path,
    landmarker_path=LANDMARKER_PATH,
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    preprocess=False
):
    """
    Varre os diretórios de frames e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos
    com o HandLandmarker do MediaPipe.

    Exporta o dataset para output_path e o log da extração para logs/<nome do csv>.txt.
    """
    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não encontrado.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(landmarker_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,  # detectar até 2 mãos por frame
    )

    features = []
    labels = []
    groups = []

    nome_csv = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(LOGS_DIR, exist_ok=True)
    feature_logs_path = f"{LOGS_DIR}/{nome_csv}.txt"
    file_ = open(feature_logs_path, 'w')
    file_.write(f"Pré-processamento: {'sim' if preprocess else 'não'}\n\n")

    frames_totais_global = 0
    frames_detectados = 0
    frames_padding_global = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for gesto in sorted(os.listdir(dataset_root_dir)):
            class_dir = os.path.join(dataset_root_dir, gesto)

            if not os.path.isdir(class_dir):
                continue

            print(f"Extraindo features do gesto {gesto}")

            videos_dir = os.listdir(class_dir)
            sequencias_classe = 0
            frames_duplicados = 0
            frames_totais = 0
            frames_padding = 0
            num_videos_padding = 0

            for video_dir in videos_dir:
                path_video = os.path.join(class_dir, video_dir)
                frame_files = os.listdir(path_video)
                # Ordenação pelo frame obrigatória para manter sequência temporal
                frames = sorted(frame_files, key=lambda x: int(x.removesuffix('.jpg')))

                if not frames:
                    continue
                frames_totais += len(frames)
                frames_totais_global += len(frames)
                # ID usado para identificar o Grupo no StratifiedGroupKFold
                video_id = f"{gesto}_{os.path.basename(video_dir)}"
                video_landmarks = []
                ultimo_valido = [0.0] * NUM_FEATURES_MAOS

                for frame in frames:
                    # Coluna informativa para o processo de Data Augmentation
                    duplicado = False
                    frame_path = os.path.join(path_video, frame)
                    try:
                        if preprocess:
                            img_bgr = preprocessImage(frame_path)
                        else:
                            img_bgr = cv2.imread(frame_path)
                        if img_bgr is None:
                            raise ValueError("não foi possível ler a imagem")
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # Detecção das landmarks
                        maos = detectar_mediapipe(landmarker, img_rgb)

                        if maos:
                            landmarks = maos_para_features(maos)
                            frames_detectados += 1
                            ultimo_valido = landmarks[:]
                        else:
                            duplicado = True
                            frames_duplicados += 1

                            # Forward-fill com o último frame válido
                            landmarks = ultimo_valido[:]

                        line = landmarks
                        line.append(1 if duplicado else 0)
                        video_landmarks.append(line)

                    except Exception as e:
                        print(f"Erro ao processar {frame_path}: {e}")

                # Back-fill em frames iniciais sem detecção utilizando a primeira mão válida encontrada
                # (a coluna duplicado é ignorada na verificação de frame vazio)
                primeira_valida = next(
                    (lm for lm in video_landmarks if any(lm[:NUM_FEATURES_MAOS])), None
                )
                if primeira_valida is not None:
                    for idx_lm in range(len(video_landmarks)):
                        if not any(video_landmarks[idx_lm][:NUM_FEATURES_MAOS]):
                            video_landmarks[idx_lm] = primeira_valida[:NUM_FEATURES_MAOS] + [1]
                        else:
                            break

                # Preenche o final do video com "frames vazios" para garantir
                # pelo menos uma amostra de cada gesto nos testes
                if len(video_landmarks) < sequence_length:
                    num_frames_padding = sequence_length - len(video_landmarks)
                    frames_padding += num_frames_padding
                    frames_padding_global += num_frames_padding
                    num_videos_padding += 1
                    video_landmarks.extend(
                        [ZEROS_FRAME_MAOS for _ in range(num_frames_padding)]
                    )

                # Geração de Amostras de SEQUENCE_LENGTH frames
                for i in range(0, len(video_landmarks) - sequence_length + 1, step):
                    features.append(video_landmarks[i:i + sequence_length])
                    labels.append(gesto)
                    groups.append(video_id)
                    sequencias_classe += 1
            print(f'Frames duplicados em {gesto}: {frames_duplicados}/{frames_totais}')
            print(f'Frames de padding em {gesto}: {frames_padding}/{frames_totais+frames_padding}')
            print(f'Vídeos com padding em {gesto}: {num_videos_padding}/{len(videos_dir)}')
            file_.write(f'Frames duplicados em {gesto}: {frames_duplicados}/{frames_totais}\n')
            file_.write(f'Frames de padding em {gesto}: {frames_padding}/{frames_totais+frames_padding}\n')
            file_.write(f'Vídeos com padding em {gesto}: {num_videos_padding}/{len(videos_dir)}\n')

            if sequencias_classe == 0:
                print(f"  AVISO: '{gesto}' gerou 0 sequências.\n"+
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis.")
                file_.write(f"  AVISO: '{gesto}' gerou 0 sequências.\n"+
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis.\n")

            else:
                print(f"  -> {sequencias_classe} sequência(s) para '{gesto}'")
                file_.write(f"  -> {sequencias_classe} sequência(s) para '{gesto}'\n")

    if not features:
        print("[ERRO] Nenhuma amostra foi gerada.")
        file_.write("[ERRO] Nenhuma amostra foi gerada.\n")
        file_.close()
        return

    def porcentagem(parte, total):
        return f"{parte}/{total} = {parte / total * 100 if total else 0.0}%"

    print(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0])})')
    print(f"\nExtração concluída! Total de {len(features)} amostras coletadas")
    print(f"Detecção de Frames: {porcentagem(frames_detectados, frames_totais_global)}")
    file_.write(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0])})\n')
    file_.write(f"\nExtração concluída! Total de {len(features)} amostras coletadas\n")
    file_.write(f"\nDetecção de Frames: {porcentagem(frames_detectados, frames_totais_global)}")
    file_.write(f"\nNúmero de Frames com Padding: {porcentagem(frames_padding_global, frames_totais_global + frames_padding_global)}")
    file_.close()

    # Exportação para CSV dos dados
    print('Exportando dataset para csv')

    coord_cols = []
    for prefixo in ("d", "e"):  # d = direita, e = esquerda
        for i in range(1, 22):
            coord_cols += [
                f"{prefixo}_x_{i}", f"{prefixo}_y_{i}",
                f"{prefixo}_z_{i}"
            ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exportando {len(labels)} amostras...")
    rows = []

    for sample_idx, (seq, label) in enumerate(zip(features, labels)):

        group_val = groups[sample_idx]
        for frame_idx, frame in enumerate(seq):
            row = [label, group_val, sample_idx, frame_idx] + list(frame)
            rows.append(row)

    all_cols = ["target", "video_id", "sample_idx", "frame_idx"] + coord_cols + ["duplicado"]
    df = pd.DataFrame(rows, columns=all_cols)

    df.to_csv(output_path, index=False)
    print(f"Dataset exportado: {output_path}")


def extrair_datasets(preprocess=False):
    """
    Extrai os datasets de treino e teste com a configuração informada, salvando em
    dataset/{treino|teste}_<configuração>.csv e logs/{treino|teste}_<configuração>.txt.
    """
    nome = nome_configuracao(preprocess)
    for split, frames_dir in (("treino", FRAMES_TREINO_DIR), ("teste", FRAMES_TESTE_DIR)):
        print(f'Gerando dataset de {split.upper()} ({nome})')
        extract_features_from_directory(
            dataset_root_dir=frames_dir,
            output_path=f"dataset/{split}_{nome}.csv",
            sequence_length=SEQUENCE_LENGTH,
            step=DEFAULT_STEP,
            preprocess=preprocess
        )


if __name__ == "__main__":
    extrair_datasets(preprocess=True)
