import multiprocessing
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import (
    DATASET_TESTE_CSV,
    DATASET_TREINO_CSV,
    DEFAULT_STEP,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    HOLISTIC_LANDMARKER_PATH,
    LOGS_DIR,
    NUM_FEATURES,
    NUM_LANDMARKS_PER_HAND,
    POSE_LANDMARKS_USADOS,
    SEQUENCE_LENGTH,
    ZEROS_FRAME,
)
from utils.constants_cv import (
    BaseOptions,
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import normalizar_holistic


def preprocessImage(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    # Padronização do tamanho de imagens
    target_width = 1280
    target_height = 720
    original_height, original_width = image.shape[:2]
    scale = min(
        target_width / original_width,
        target_height / original_height
    )
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    new_image = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    new_image[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    # Aplicação do Blur
    blurred_image = cv2.bilateralFilter(
        new_image, d=5, 
        sigmaColor=75,
        sigmaSpace=75
    )

    # Aplicação do Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp_image = cv2.filter2D(blurred_image, -1, kernel)
    return sharp_image

    # Retornar a imagem

# Processos paralelos da extração
NUM_PROCESSOS = max(1, (os.cpu_count() or 1))

# Configuração do Holistic em cada processo da extração paralela
_holistic_options = None


def _iniciar_processo(holistic_path):
    """Inicializador dos processos paralelos: configura o Holistic no modo IMAGE."""
    global _holistic_options
    _holistic_options = HolisticLandmarkerOptions(
        base_options=BaseOptions(holistic_path),
        running_mode=VisionRunningMode.IMAGE,
    )


def _processar_frame(args):
    """
    Detecta pose e mãos em um único frame com o Holistic (executado nos processos paralelos).
    Retorna as features normalizadas do frame (None quando a pose não é detectada), se a mão
    direita e a esquerda foram detectadas e a mensagem de erro (None se o frame foi processado).
    """
    frame_path, preprocess = args
    try:
        if preprocess:
            img_bgr = preprocessImage(frame_path)
        else:
            img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            raise ValueError("não foi possível ler a imagem")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        altura, largura = img_rgb.shape[:2]

        # Um Holistic novo por frame: mesmo no modo IMAGE o Holistic guarda estado entre as
        # chamadas (a pose de um frame muda conforme os frames processados antes dele), então
        # recriá-lo garante que cada frame seja processado de forma independente
        with HolisticLandmarker.create_from_options(_holistic_options) as holistic:
            resultado = holistic.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))

        landmarks = normalizar_holistic(resultado, largura, altura)
        return landmarks, bool(resultado.right_hand_landmarks), bool(resultado.left_hand_landmarks), None

    except Exception as e:
        return None, False, False, f"Erro ao processar {frame_path}: {e}"


def colunas_coordenadas():
    """
    Nome das colunas de coordenadas do CSV, na ordem das features:
    d = mão direita, e = mão esquerda (landmarks 1 a 21, x, y e z),
    p = pose (pelo nome do ponto em POSE_LANDMARKS_USADOS, apenas x e y).
    """
    colunas = []
    for prefixo in ("d", "e"):
        for i in range(1, NUM_LANDMARKS_PER_HAND + 1):
            colunas += [f"{prefixo}_x_{i}", f"{prefixo}_y_{i}", f"{prefixo}_z_{i}"]
    for nome in POSE_LANDMARKS_USADOS.values():
        colunas += [f"p_x_{nome}", f"p_y_{nome}"]
    return colunas


def extract_features_from_directory(
    dataset_root_dir,
    output_path,
    holistic_path=HOLISTIC_LANDMARKER_PATH,
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    preprocess=False,
    num_processos=NUM_PROCESSOS
):
    """
    Varre os diretórios de frames e extrai, com o MediaPipe Holistic, sequências 3D de AMBAS as mãos e
    dos pontos do Pose em POSE_LANDMARKS_USADOS (ombros, cotovelos, pulsos e quadris), normalizadas em relação ao corpo
    (ver normalizar_holistic).

    Features no formato: (amostras, frames, NUM_FEATURES).

    A detecção roda em num_processos processos paralelos, frame a frame (cada frame é processado
    de forma independente, sem tracking entre os frames do vídeo).
    Frames sem pose detectada são preenchidos com o último frame válido (coluna duplicado = 1).
    Exporta o dataset para output_path e o log da extração para logs/<nome do csv>.txt.
    """
    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não encontrado.")
        return
    if not os.path.exists(holistic_path):
        raise FileNotFoundError(f"Modelo do Holistic '{holistic_path}' não encontrado.")

    # Estrutura do dataset: gesto -> vídeos -> frames ordenados (ordem temporal obrigatória)
    estrutura = []
    for gesto in sorted(os.listdir(dataset_root_dir)):
        class_dir = os.path.join(dataset_root_dir, gesto)
        if not os.path.isdir(class_dir):
            continue
        videos = []
        for video_dir in sorted(os.listdir(class_dir)):
            path_video = os.path.join(class_dir, video_dir)
            frames = sorted(os.listdir(path_video), key=lambda x: int(x.removesuffix('.jpg')))
            videos.append((video_dir, path_video, frames))
        estrutura.append((gesto, videos))
    tarefas = [
        (os.path.join(path_video, frame), preprocess)
        for _, videos in estrutura for _, path_video, frames in videos for frame in frames
    ]

    features = []
    labels = []
    groups = []

    nome_csv = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(LOGS_DIR, exist_ok=True)
    feature_logs_path = f"{LOGS_DIR}/{nome_csv}.txt"
    file_ = open(feature_logs_path, 'w')
    file_.write(f"Extrator: MediaPipe Holistic ({holistic_path})\n")
    file_.write("Normalização: relativa ao corpo (origem no centro dos ombros, escala pela largura dos ombros)\n")
    file_.write(f"Pré-processamento: {'sim' if preprocess else 'não'}\n\n")

    frames_totais_global = 0
    frames_padding_global = 0
    frames_pose = 0
    frames_mao_direita = 0
    frames_mao_esquerda = 0

    print(f"Processos paralelos: {num_processos}")
    with multiprocessing.Pool(
        num_processos,
        initializer=_iniciar_processo,
        initargs=(holistic_path,)
    ) as pool, tqdm(total=len(tarefas), unit="frame", desc=nome_csv) as barra:
        # imap mantém a ordem dos frames, então as sequências são montadas na ordem do dataset
        resultados = pool.imap(_processar_frame, tarefas, chunksize=16)

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

                for _ in frames:
                    landmarks_frame, tem_mao_direita, tem_mao_esquerda, erro = next(resultados)
                    barra.update(1)
                    if erro is not None:
                        tqdm.write(erro)
                        continue

                    # Coluna informativa para o processo de Data Augmentation
                    duplicado = False
                    if landmarks_frame is not None:
                        landmarks = landmarks_frame
                        ultimo_valido = landmarks[:]
                        frames_pose += 1
                        frames_mao_direita += tem_mao_direita
                        frames_mao_esquerda += tem_mao_esquerda
                    else:
                        duplicado = True
                        frames_duplicados += 1

                        # Forward-fill com o último frame válido
                        landmarks = ultimo_valido[:]

                    line = landmarks
                    line.append(1 if duplicado else 0)
                    video_landmarks.append(line)

                # Back-fill em frames iniciais sem detecção utilizando o primeiro frame válido encontrado
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
                        [ZEROS_FRAME + [0] for _ in range(num_frames_padding)]
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

    print(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0]) - 1})')
    print(f"\nExtração concluída! Total de {len(features)} amostras coletadas")
    print(f"Frames com pose detectada: {porcentagem(frames_pose, frames_totais_global)}")
    file_.write(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0]) - 1})\n')
    file_.write(f"\nExtração concluída! Total de {len(features)} amostras coletadas\n")
    file_.write(f"\nFrames com pose detectada: {porcentagem(frames_pose, frames_totais_global)}")
    file_.write(f"\nFrames com mão direita detectada: {porcentagem(frames_mao_direita, frames_totais_global)}")
    file_.write(f"\nFrames com mão esquerda detectada: {porcentagem(frames_mao_esquerda, frames_totais_global)}")
    file_.write(f"\nNúmero de Frames com Padding: {porcentagem(frames_padding_global, frames_totais_global + frames_padding_global)}")
    file_.close()

    # Exportação para CSV dos dados
    print('Exportando dataset para csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exportando {len(labels)} amostras...")
    rows = []

    for sample_idx, (seq, label) in enumerate(zip(features, labels)):

        group_val = groups[sample_idx]
        for frame_idx, frame in enumerate(seq):
            row = [label, group_val, sample_idx, frame_idx] + list(frame)
            rows.append(row)

    all_cols = ["target", "video_id", "sample_idx", "frame_idx"] + colunas_coordenadas() + ["duplicado"]
    df = pd.DataFrame(rows, columns=all_cols)

    df.to_csv(output_path, index=False)
    print(f"Dataset exportado: {output_path}")

def import_from_csv(filepath: str):
    """
    Carrega o dataset a partir de um CSV no formato 3D para o modelo LSTM: (amostras, frames, features).
    Retorna 3 arrays features (amostras do dataset), labels (labels de cada amostra), 
    groups (video ao qual pertence a amostra).
    """
    df = pd.read_csv(filepath)

    feature_cols = [col for col in df.columns if col not in["target", "video_id", "frame_idx", "sample_idx", "duplicado"]]
    unique_samples = df["sample_idx"].unique()
    num_samples = len(unique_samples)
    num_frames = df["frame_idx"].nunique()
    num_features = len(feature_cols)

    features = np.zeros((num_samples, num_frames, num_features))
    labels = []
    groups = []


    for i, (sample_idx, df_sample) in enumerate(df.groupby("sample_idx", sort=False)):
        df_sample = df_sample.sort_values(by="frame_idx")

        landmarks_matrix = df_sample[feature_cols].values
        t_real = landmarks_matrix.shape[0]
        features[i, :t_real, :] = landmarks_matrix
        sample_label = df_sample["target"].iloc[0]
        labels.append(sample_label)

        groups.append(df_sample["video_id"].iloc[0])

    labels = np.array(labels)
    groups = np.array(groups)

    return features.astype(np.float32), labels, groups


if __name__ == "__main__":
    print('Gerando dataset de TREINO')
    extract_features_from_directory(
        dataset_root_dir=FRAMES_TREINO_DIR,
        output_path=DATASET_TREINO_CSV,
        sequence_length=SEQUENCE_LENGTH,
        step=DEFAULT_STEP,
        preprocess=True
    )

    print('Gerando dataset de TESTE')
    extract_features_from_directory(
        dataset_root_dir=FRAMES_TESTE_DIR,
        output_path=DATASET_TESTE_CSV,
        sequence_length=SEQUENCE_LENGTH,
        step=DEFAULT_STEP,
        preprocess=True
    )
