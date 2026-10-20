import os
from collections import namedtuple
from contextlib import nullcontext

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from feature_extraction import preprocessImage
from utils.constants import (
    DEFAULT_STEP,
    DIST_MAX_MESMA_MAO,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    LANDMARKER_PATH,
    LOGS_DIR,
    NUM_FEATURES,
    NUM_HANDS,
    NUM_LANDMARKS_PER_HAND,
    SEQUENCE_LENGTH,
    YOLO_CONFIANCA,
    YOLO_CONFIANCA_KPT,
    YOLO_HAND_MODEL_PATH,
    ZEROS_FRAME,
    ZEROS_MAO,
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import normalizar_mao

MODOS = ("mediapipe", "yolo", "both", "fallback")
EXTRATORES = ("mediapipe", "yolo")

# Landmarks necessários para a normalização (pulso, base indicador, base mínimo)
LANDMARKS_REFERENCIA = (0, 5, 17)

# Ponto no mesmo formato dos landmarks do MediaPipe, para reutilizar normalizar_mao
Ponto = namedtuple("Ponto", ["x", "y", "z"])


def nome_configuracao(mode, priority="mediapipe", preprocess=False):
    """Sufixo usado nos nomes do CSV e do log, para cada configuração não sobrescrever as outras."""
    nome = f"fallback_{priority}" if mode == "fallback" else mode
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


def detectar_yolo(modelo, img_bgr):
    """
    Retorna uma lista com até NUM_HANDS mãos detectadas pelo YOLO-Pose (arrays (21, 3)),
    ordenadas pela confiança da detecção. O YOLO não estima profundidade, então Z = 0.
    """
    result = modelo(img_bgr, conf=YOLO_CONFIANCA, verbose=False)[0]
    if result.keypoints is None or len(result.boxes) == 0:
        return []

    keypoints_xy = result.keypoints.xyn.cpu().numpy()  # (n, 21, 2)
    if keypoints_xy.shape[1] != NUM_LANDMARKS_PER_HAND:
        raise ValueError(
            f"O modelo YOLO retornou {keypoints_xy.shape[1]} keypoints, esperado "
            f"{NUM_LANDMARKS_PER_HAND}. Utilize um modelo pose treinado no dataset hand-keypoints."
        )
    keypoints_conf = result.keypoints.conf
    keypoints_conf = (
        keypoints_conf.cpu().numpy() if keypoints_conf is not None
        else np.ones(keypoints_xy.shape[:2])
    )
    ordem = np.argsort(-result.boxes.conf.cpu().numpy())

    maos = []
    for idx in ordem:
        # Descarta mãos sem os pontos de referência confiáveis, pois a normalização depende deles
        if np.any(keypoints_conf[idx, list(LANDMARKS_REFERENCIA)] < YOLO_CONFIANCA_KPT):
            continue
        mao = np.zeros((NUM_LANDMARKS_PER_HAND, 3))
        mao[:, :2] = keypoints_xy[idx]
        maos.append(mao)
        if len(maos) == NUM_HANDS:
            break
    return maos


def atribuir_lateralidade(maos_yolo, referencias):
    """
    O YOLO não informa a lateralidade das mãos, então cada mão é associada a "Right" ou "Left":

    1. Pela mão de referência mais próxima (distância média em XY entre os 21 landmarks),
       desde que abaixo de DIST_MAX_MESMA_MAO. As referências são as mãos do MediaPipe no
       mesmo frame ou a última posição conhecida de cada mão no vídeo.
    2. As mãos restantes ocupam os lados livres pela posição horizontal: no dataset o MediaPipe
       rotula como "Right" a mão com menor X na imagem.
    """
    maos = {}
    livres = list(range(len(maos_yolo)))

    pares = []
    for idx, mao in enumerate(maos_yolo):
        for label, ref in referencias.items():
            dist = np.mean(np.linalg.norm(mao[:, :2] - ref[:, :2], axis=1))
            if dist < DIST_MAX_MESMA_MAO:
                pares.append((dist, idx, label))

    for _, idx, label in sorted(pares):
        if idx in livres and label not in maos:
            maos[label] = maos_yolo[idx]
            livres.remove(idx)

    # Ordena as mãos restantes pelo X do pulso (menor X -> "Right")
    livres.sort(key=lambda i: maos_yolo[i][0, 0])
    labels_livres = [label for label in ("Right", "Left") if label not in maos]
    if len(livres) == 1 and len(labels_livres) == 2:
        labels_livres = ["Right"] if maos_yolo[livres[0]][0, 0] < 0.5 else ["Left"]
    for idx, label in zip(livres, labels_livres):
        maos[label] = maos_yolo[idx]

    return maos


def combinar_deteccoes(maos_mp, maos_yolo):
    """
    Modo both: tira a média de cada landmark respectivo quando as duas ferramentas detectam a
    mesma mão. Como o YOLO não estima Z, a profundidade é mantida a do MediaPipe.
    Mãos detectadas por apenas uma das ferramentas são usadas diretamente.
    """
    maos = {}
    em_ambos = 0
    for label in ("Right", "Left"):
        mao_mp = maos_mp.get(label)
        mao_yolo = maos_yolo.get(label)
        if mao_mp is not None and mao_yolo is not None:
            mao = mao_mp.copy()
            mao[:, :2] = (mao_mp[:, :2] + mao_yolo[:, :2]) / 2
            maos[label] = mao
            em_ambos += 1
        elif mao_mp is not None:
            maos[label] = mao_mp
        elif mao_yolo is not None:
            maos[label] = mao_yolo
    return maos, em_ambos > 0


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
    mode="mediapipe",
    priority="mediapipe",
    landmarker_path=LANDMARKER_PATH,
    yolo_model_path=YOLO_HAND_MODEL_PATH,
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    preprocess=False
):
    """
    Varre os diretórios de frames e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos,
    utilizando o extrator definido em mode:
        - mediapipe: apenas MediaPipe
        - yolo: apenas YOLO-Pose
        - both: MediaPipe e YOLO, com a média de cada landmark respectivo
        - fallback: usa o extrator definido em priority e, caso não detecte nenhuma mão,
          tenta com o outro extrator (priority é ignorado nos demais modos)

    Exporta o dataset para output_path e o log da extração para logs/<nome do csv>.txt.
    """
    if mode not in MODOS:
        raise ValueError(f"mode inválido: '{mode}'. Valores aceitos: {MODOS}")
    if mode == "fallback" and priority not in EXTRATORES:
        raise ValueError(f"priority inválido: '{priority}'. Valores aceitos: {EXTRATORES}")

    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não encontrado.")
        return

    usa_mediapipe = mode in ("mediapipe", "both", "fallback")
    usa_yolo = mode in ("yolo", "both", "fallback")
    secundario = "yolo" if priority == "mediapipe" else "mediapipe"

    yolo = None
    if usa_yolo:
        # Import local para o modo mediapipe não depender do ultralytics
        from ultralytics import YOLO
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"Modelo YOLO '{yolo_model_path}' não encontrado.")
        yolo = YOLO(yolo_model_path)

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
    file_.write(f"Modo: {mode}" + (f" (prioridade: {priority})" if mode == "fallback" else "") + "\n")
    file_.write(f"Pré-processamento: {'sim' if preprocess else 'não'}\n\n")

    frames_totais_global = 0
    frames_detectados = 0
    frames_padding_global = 0
    # Frames em que cada extrator detectou pelo menos uma mão
    deteccoes_mediapipe = 0
    deteccoes_yolo = 0
    frames_media = 0  # modo both: frames com pelo menos uma mão detectada por ambos
    frames_secundario = 0  # modo fallback: frames resolvidos pelo extrator secundário

    landmarker_ctx = HandLandmarker.create_from_options(options) if usa_mediapipe else nullcontext()
    with landmarker_ctx as landmarker:
        def detectar(extrator, img_bgr, img_rgb, referencias):
            nonlocal deteccoes_mediapipe, deteccoes_yolo
            if extrator == "mediapipe":
                maos = detectar_mediapipe(landmarker, img_rgb)
                deteccoes_mediapipe += bool(maos)
            else:
                maos = atribuir_lateralidade(detectar_yolo(yolo, img_bgr), referencias)
                deteccoes_yolo += bool(maos)
            return maos

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
                ultimo_valido = [0.0] * NUM_FEATURES
                # Última posição conhecida de cada mão no vídeo, usada para a lateralidade do YOLO
                ultimas_maos = {}

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
                        if mode == "both":
                            maos_mp = detectar("mediapipe", img_bgr, img_rgb, ultimas_maos)
                            maos_yolo = detectar("yolo", img_bgr, img_rgb, {**ultimas_maos, **maos_mp})
                            maos, usou_media = combinar_deteccoes(maos_mp, maos_yolo)
                            frames_media += usou_media
                        elif mode == "fallback":
                            maos = detectar(priority, img_bgr, img_rgb, ultimas_maos)
                            if not maos:
                                maos = detectar(secundario, img_bgr, img_rgb, ultimas_maos)
                                frames_secundario += bool(maos)
                        else:
                            maos = detectar(mode, img_bgr, img_rgb, ultimas_maos)

                        if maos:
                            landmarks = maos_para_features(maos)
                            frames_detectados += 1
                            ultimo_valido = landmarks[:]
                            ultimas_maos.update(maos)
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
                    (lm for lm in video_landmarks if any(lm[:NUM_FEATURES])), None
                )
                if primeira_valida is not None:
                    for idx_lm in range(len(video_landmarks)):
                        if not any(video_landmarks[idx_lm][:NUM_FEATURES]):
                            video_landmarks[idx_lm] = primeira_valida[:NUM_FEATURES] + [1]
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
                        [ZEROS_FRAME for _ in range(num_frames_padding)]
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
    if usa_mediapipe:
        file_.write(f"\nDetecções do MediaPipe: {porcentagem(deteccoes_mediapipe, frames_totais_global)}")
    if usa_yolo:
        file_.write(f"\nDetecções do YOLO: {porcentagem(deteccoes_yolo, frames_totais_global)}")
    if mode == "both":
        file_.write(f"\nFrames com média entre MediaPipe e YOLO: {porcentagem(frames_media, frames_totais_global)}")
    if mode == "fallback":
        file_.write(f"\nFrames recuperados pelo {secundario}: {porcentagem(frames_secundario, frames_totais_global)}")
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


def extrair_datasets(mode, priority="mediapipe", preprocess=False):
    """
    Extrai os datasets de treino e teste com a configuração informada, salvando em
    dataset/{treino|teste}_<configuração>.csv e logs/{treino|teste}_<configuração>.txt.
    """
    nome = nome_configuracao(mode, priority, preprocess)
    for split, frames_dir in (("treino", FRAMES_TREINO_DIR), ("teste", FRAMES_TESTE_DIR)):
        print(f'Gerando dataset de {split.upper()} ({nome})')
        extract_features_from_directory(
            dataset_root_dir=frames_dir,
            output_path=f"dataset/{split}_{nome}.csv",
            mode=mode,
            priority=priority,
            sequence_length=SEQUENCE_LENGTH,
            step=DEFAULT_STEP,
            preprocess=preprocess
        )


if __name__ == "__main__":
    CONFIGURACOES = [
        {"mode": "mediapipe"},
        {"mode": "yolo"},
        {"mode": "both"},
        {"mode": "fallback", "priority": "mediapipe"},
        {"mode": "fallback", "priority": "yolo"},
    ]

    for config in CONFIGURACOES:
        extrair_datasets(**config, preprocess=True)
