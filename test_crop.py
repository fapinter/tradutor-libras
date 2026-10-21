import os
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extraction import preprocessImage
from tests_extraction import maos_para_features
from utils.constants import (
    CROP_ESCALA_OMBROS,
    DEFAULT_STEP,
    DIST_MAX_PULSO_POSE,
    DIST_MESMA_MAO_CROP,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    LANDMARKER_PATH,
    LOGS_DIR,
    NUM_FEATURES,
    NUM_HANDS,
    POSE_LANDMARKER_PATH,
    SEQUENCE_LENGTH,
    ZEROS_FRAME,
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    VisionRunningMode,
)

MODOS = ("separado", "unico")

# Landmarks do Pose de cada mão da pessoa: pulso, mínimo, indicador, polegar.
# A mão direita da pessoa corresponde ao rótulo "Right" do MediaPipe Hands neste dataset.
POSE_MAOS = {"Right": (16, 18, 20, 22), "Left": (15, 17, 19, 21)}
POSE_OMBROS = (11, 12)

# Processos paralelos da extração (deixa 2 núcleos livres para o sistema)
NUM_PROCESSOS = max(1, (os.cpu_count() or 1) - 2)


def nome_configuracao(mode, preprocess=False):
    """Sufixo usado nos nomes do CSV e do log, para cada modo não sobrescrever o outro."""
    nome = f"crop_{mode}"
    if preprocess:
        nome += "_preprocess"
    return nome


def detectar_pose(pose_landmarker, img_rgb):
    """
    Retorna os pontos de cada mão estimados pelo Pose, em pixels ({"Right"|"Left": array (4, 2)}),
    e a largura dos ombros em pixels. Os pontos podem estar fora da imagem.
    Retorna (None, None) se nenhuma pessoa for detectada.
    """
    results = pose_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not results.pose_landmarks:
        return None, None

    lms = results.pose_landmarks[0]
    h, w = img_rgb.shape[:2]
    maos = {
        label: np.array([[lms[i].x * w, lms[i].y * h] for i in idx])
        for label, idx in POSE_MAOS.items()
    }
    ombro_d, ombro_e = (lms[i] for i in POSE_OMBROS)
    ombros = np.hypot((ombro_d.x - ombro_e.x) * w, (ombro_d.y - ombro_e.y) * h)
    return maos, max(ombros, 1.0)


def caixa_mao(pontos, ombros, largura, altura):
    """
    Quadrado (x0, y0, lado), em pixels, centrado nos pontos da mão do Pose.
    Retorna None se nenhum ponto da mão estiver dentro da imagem (mão fora do quadro).
    """
    dentro = (
        (pontos[:, 0] >= 0) & (pontos[:, 0] < largura) &
        (pontos[:, 1] >= 0) & (pontos[:, 1] < altura)
    )
    if not dentro.any():
        return None
    lado = CROP_ESCALA_OMBROS * ombros
    cx, cy = pontos.mean(0)
    return (cx - lado / 2, cy - lado / 2, lado)


def unir_caixas(caixas):
    """Menor quadrado que contém todas as caixas."""
    x0 = min(c[0] for c in caixas)
    y0 = min(c[1] for c in caixas)
    x1 = max(c[0] + c[2] for c in caixas)
    y1 = max(c[1] + c[2] for c in caixas)
    lado = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - lado / 2, cy - lado / 2, lado)


def recortar(img, caixa):
    """Recorte quadrado da imagem. Áreas fora da imagem são preenchidas com preto para manter a proporção."""
    x0, y0, lado = (int(round(v)) for v in caixa)
    lado = max(lado, 1)
    h, w = img.shape[:2]
    crop = np.zeros((lado, lado, 3), dtype=np.uint8)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + lado, w), min(y0 + lado, h)
    if xb > xa and yb > ya:
        crop[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return crop, (x0, y0, lado)


def detectar_no_recorte(landmarker, img_rgb, caixa):
    """
    Aplica o HandLandmarker no recorte e converte os landmarks para coordenadas normalizadas da
    imagem inteira, mantendo o formato do dataset atual. Retorna uma lista de arrays (21, 3).
    """
    crop, (x0, y0, lado) = recortar(img_rgb, caixa)
    results = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))

    h, w = img_rgb.shape[:2]
    maos = []
    for hand_lms in results.hand_landmarks:
        mao = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
        mao[:, 0] = (mao[:, 0] * lado + x0) / w
        mao[:, 1] = (mao[:, 1] * lado + y0) / h
        # Z do MediaPipe está na escala do X do recorte -> converte para a escala do X da imagem inteira
        mao[:, 2] = mao[:, 2] * lado / w
        maos.append(mao)
    return maos


def atribuir_maos(deteccoes, pose_maos, ombros, largura, altura):
    """
    Associa cada mão detectada à mão do Pose com o pulso mais próximo (cada mão uma única vez),
    descartando detecções com pulso a mais de DIST_MAX_PULSO_POSE larguras de ombro.
    A lateralidade vem do Pose, e não do classificador de lateralidade do MediaPipe Hands.
    """
    pares = []
    for idx, mao in enumerate(deteccoes):
        pulso = mao[0, :2] * [largura, altura]
        for label, pontos in pose_maos.items():
            dist = np.linalg.norm(pulso - pontos[0]) / ombros
            if dist < DIST_MAX_PULSO_POSE:
                pares.append((dist, idx, label))

    maos = {}
    usadas = set()
    for _, idx, label in sorted(pares):
        if idx in usadas or label in maos:
            continue
        maos[label] = deteccoes[idx]
        usadas.add(idx)
    return maos


def mesma_mao(a, b, ombros, largura, altura):
    """Duas detecções são a mesma mão se a distância média entre os landmarks for pequena."""
    dist = np.linalg.norm((a[:, :2] - b[:, :2]) * [largura, altura], axis=1).mean()
    return dist / ombros < DIST_MESMA_MAO_CROP


def extrair_separado(landmarker, img_rgb, pose_maos, ombros):
    """
    Um recorte por mão visível. As detecções de todos os recortes são reunidas, as repetidas
    (mesma mão vista em dois recortes sobrepostos) são removidas e só então associadas às mãos do Pose,
    pois com as mãos cruzadas o recorte de uma mão pode encontrar apenas a outra.
    Retorna as mãos detectadas e o número de recortes feitos.
    """
    h, w = img_rgb.shape[:2]
    deteccoes = []
    recortes = 0
    for pontos in pose_maos.values():
        caixa = caixa_mao(pontos, ombros, w, h)
        if caixa is None:
            continue
        recortes += 1
        for mao in detectar_no_recorte(landmarker, img_rgb, caixa):
            if not any(mesma_mao(mao, outra, ombros, w, h) for outra in deteccoes):
                deteccoes.append(mao)
    return atribuir_maos(deteccoes, pose_maos, ombros, w, h), recortes


def extrair_unico(landmarker, img_rgb, pose_maos, ombros):
    """
    Um único recorte quadrado contendo as mãos visíveis.
    Retorna as mãos detectadas e o número de mãos visíveis no recorte.
    """
    h, w = img_rgb.shape[:2]
    caixas = [c for c in (caixa_mao(p, ombros, w, h) for p in pose_maos.values()) if c is not None]
    if not caixas:
        return {}, 0
    deteccoes = detectar_no_recorte(landmarker, img_rgb, unir_caixas(caixas))
    return atribuir_maos(deteccoes, pose_maos, ombros, w, h), len(caixas)


# Detectores de cada processo da extração paralela (criados uma única vez por processo)
_landmarker = None
_pose_landmarker = None


def _iniciar_processo(landmarker_path, pose_landmarker_path):
    """Inicializador dos processos paralelos: cada processo tem o seu próprio HandLandmarker e Pose."""
    global _landmarker, _pose_landmarker
    _landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(landmarker_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,  # o recorte pode conter as duas mãos
    ))
    _pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(pose_landmarker_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    ))


def _processar_video(args):
    """
    Detecta as mãos em todos os frames de um vídeo (executado nos processos paralelos).
    Retorna, na ordem dos frames, as features de cada frame (None quando nenhuma mão é detectada),
    as estatísticas do vídeo e as mensagens dos frames que não puderam ser processados.
    """
    path_video, frames, mode, preprocess = args
    extrair = extrair_separado if mode == "separado" else extrair_unico

    deteccoes = []
    stats = {"sem_pose": 0, "visiveis": 0, "detectadas": 0}
    erros = []
    for frame in frames:
        frame_path = os.path.join(path_video, frame)
        try:
            if preprocess:
                img_bgr = preprocessImage(frame_path)
            else:
                img_bgr = cv2.imread(frame_path)
            if img_bgr is None:
                raise ValueError("não foi possível ler a imagem")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Localização das mãos pelo Pose e detecção das landmarks nos recortes
            pose_maos, ombros = detectar_pose(_pose_landmarker, img_rgb)
            if pose_maos is None:
                stats["sem_pose"] += 1
                maos = {}
            else:
                maos, visiveis = extrair(_landmarker, img_rgb, pose_maos, ombros)
                stats["visiveis"] += visiveis
                stats["detectadas"] += len(maos)

            deteccoes.append(maos_para_features(maos) if maos else None)

        except Exception as e:
            erros.append(f"Erro ao processar {frame_path}: {e}")
    return deteccoes, stats, erros


def extract_features_from_directory(
    dataset_root_dir,
    output_path,
    mode="separado",
    landmarker_path=LANDMARKER_PATH,
    pose_landmarker_path=POSE_LANDMARKER_PATH,
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    preprocess=False,
    num_processos=NUM_PROCESSOS
):
    """
    Varre os diretórios de frames e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos,
    aplicando o HandLandmarker em recortes das mãos localizadas pelo MediaPipe Pose:
        - separado: um recorte quadrado por mão, com as detecções unidas em um único vetor
        - unico: um único recorte quadrado contendo as duas mãos

    A detecção roda em num_processos processos paralelos, um vídeo por vez em cada processo.
    Exporta o dataset para output_path e o log da extração para logs/<nome do csv>.txt.
    """
    if mode not in MODOS:
        raise ValueError(f"mode inválido: '{mode}'. Valores aceitos: {MODOS}")

    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não encontrado.")
        return

    # Estrutura do dataset: gesto -> vídeos -> frames ordenados (ordem temporal obrigatória)
    estrutura = []
    for gesto in sorted(os.listdir(dataset_root_dir)):
        class_dir = os.path.join(dataset_root_dir, gesto)
        if not os.path.isdir(class_dir):
            continue
        videos = []
        for video_dir in os.listdir(class_dir):
            path_video = os.path.join(class_dir, video_dir)
            frames = sorted(os.listdir(path_video), key=lambda x: int(x.removesuffix('.jpg')))
            videos.append((video_dir, path_video, frames))
        estrutura.append((gesto, videos))
    tarefas = [
        (path_video, frames, mode, preprocess)
        for _, videos in estrutura for _, path_video, frames in videos if frames
    ]

    features = []
    labels = []
    groups = []

    nome_csv = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(LOGS_DIR, exist_ok=True)
    feature_logs_path = f"{LOGS_DIR}/{nome_csv}.txt"
    file_ = open(feature_logs_path, 'w')
    file_.write(f"Modo: crop {mode} (escala do recorte: {CROP_ESCALA_OMBROS} x largura dos ombros)\n")
    file_.write(f"Pré-processamento: {'sim' if preprocess else 'não'}\n\n")

    frames_totais_global = 0
    frames_detectados = 0
    frames_padding_global = 0
    frames_sem_pose = 0
    maos_visiveis = 0  # mãos com pelo menos um ponto do Pose dentro da imagem
    maos_detectadas = 0

    print(f"Processos paralelos: {num_processos}")
    with Pool(
        num_processos,
        initializer=_iniciar_processo,
        initargs=(landmarker_path, pose_landmarker_path)
    ) as pool, tqdm(total=sum(len(t[1]) for t in tarefas), unit="frame", desc=nome_csv) as barra:
        # imap mantém a ordem dos vídeos, então as sequências são montadas como na versão sequencial
        resultados = pool.imap(_processar_video, tarefas)

        for gesto, videos in estrutura:
            tqdm.write(f"Extraindo features do gesto {gesto}")

            sequencias_classe = 0
            frames_duplicados = 0
            frames_totais = 0
            frames_padding = 0
            num_videos_padding = 0

            for video_dir, path_video, frames in videos:
                if not frames:
                    continue
                frames_totais += len(frames)
                frames_totais_global += len(frames)
                # ID usado para identificar o Grupo no StratifiedGroupKFold
                video_id = f"{gesto}_{os.path.basename(video_dir)}"
                video_landmarks = []
                ultimo_valido = [0.0] * NUM_FEATURES

                deteccoes, stats, erros = next(resultados)
                barra.update(len(frames))
                for erro in erros:
                    tqdm.write(erro)
                frames_sem_pose += stats["sem_pose"]
                maos_visiveis += stats["visiveis"]
                maos_detectadas += stats["detectadas"]

                for landmarks_frame in deteccoes:
                    # Coluna informativa para o processo de Data Augmentation
                    duplicado = False
                    if landmarks_frame is not None:
                        landmarks = landmarks_frame
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
            tqdm.write(f'Frames duplicados em {gesto}: {frames_duplicados}/{frames_totais}')
            tqdm.write(f'Frames de padding em {gesto}: {frames_padding}/{frames_totais+frames_padding}')
            tqdm.write(f'Vídeos com padding em {gesto}: {num_videos_padding}/{len(videos)}')
            file_.write(f'Frames duplicados em {gesto}: {frames_duplicados}/{frames_totais}\n')
            file_.write(f'Frames de padding em {gesto}: {frames_padding}/{frames_totais+frames_padding}\n')
            file_.write(f'Vídeos com padding em {gesto}: {num_videos_padding}/{len(videos)}\n')

            if sequencias_classe == 0:
                tqdm.write(f"  AVISO: '{gesto}' gerou 0 sequências.\n"+
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis.")
                file_.write(f"  AVISO: '{gesto}' gerou 0 sequências.\n"+
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis.\n")

            else:
                tqdm.write(f"  -> {sequencias_classe} sequência(s) para '{gesto}'")
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
    file_.write(f"\nFrames sem pessoa detectada pelo Pose: {porcentagem(frames_sem_pose, frames_totais_global)}")
    file_.write(f"\nMãos detectadas entre as mãos dentro da imagem: {porcentagem(maos_detectadas, maos_visiveis)}")
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


def extrair_datasets(mode, preprocess=False):
    """
    Extrai os datasets de treino e teste com o modo de recorte informado, salvando em
    dataset/{treino|teste}_<configuração>.csv e logs/{treino|teste}_<configuração>.txt.
    """
    nome = nome_configuracao(mode, preprocess)
    for split, frames_dir in (("treino", FRAMES_TREINO_DIR), ("teste", FRAMES_TESTE_DIR)):
        print(f'Gerando dataset de {split.upper()} ({nome})')
        extract_features_from_directory(
            dataset_root_dir=frames_dir,
            output_path=f"dataset/{split}_{nome}.csv",
            mode=mode,
            sequence_length=SEQUENCE_LENGTH,
            step=DEFAULT_STEP,
            preprocess=preprocess
        )


if __name__ == "__main__":
    for mode in MODOS:
        extrair_datasets(mode, preprocess=True)
