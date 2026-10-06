import os
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2

from utils.constants import (
    DATASET_TESTE_CSV,
    DATASET_TREINO_CSV,
    DEFAULT_STEP,
    FRAMES_TESTE_DIR,
    FRAMES_TREINO_DIR,
    LANDMARKER_PATH,
    NUM_FEATURES,
    NUM_HANDS,
    SEQUENCE_LENGTH,
    ZEROS_FRAME,
    LOGS_DIR
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import extrair_ambas_maos


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

def extract_features_from_directory(
    dataset_root_dir,
    output_path,
    landmarker_path=LANDMARKER_PATH,
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    preprocess=False
):
    """
    Varre os diretórios de frames e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos para o LSTM.

    Retorna features 3D no formato: (amostras, frames, 126_features).

    Retorna 3 listas (features, labels, grupos (identificador de videos para evitar vazamento de dados))
    """
    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não encontrado.")
        return ([], [], [])

    features = []
    labels = []
    groups = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(landmarker_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,  # detectar até 2 mãos por frame
    )

    is_train = True
    if not str(os.path.basename(output_path)).startswith('treino'):
        is_train = False
    
    if preprocess:
        feature_logs_path = f"{LOGS_DIR}/{'treino' if is_train else 'teste'}_preprocess.txt"
    else:
        feature_logs_path = f'{LOGS_DIR}/{'treino' if is_train else 'teste'}.txt'

    if not os.path.exists(feature_logs_path):
        file_ = open(feature_logs_path, 'x')
    else:
        file_ = open(feature_logs_path, 'w')
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
                ultimo_valido = [0.0] * NUM_FEATURES

                for frame in frames:
                    # Coluna informativa para o processo de Data Augmentation
                    duplicado = False
                    frame_path = os.path.join(path_video, frame)
                    try:
                        if preprocess:
                            img = preprocessImage(frame_path)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=img_rgb
                            )
                        else:
                            mp_image = mp.Image.create_from_file(frame_path)

                        # Detecção das landmarks
                        results = landmarker.detect(mp_image)

                        if results.hand_landmarks:
                            landmarks = extrair_ambas_maos(
                                results.hand_landmarks,
                                results.handedness,
                            )
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

                # Validar se é necessário
                primeira_valida = next((lm for lm in video_landmarks if any(lm)), None)
                if primeira_valida is not None:
                    # Analisa cada frame
                    for idx_lm in range(len(video_landmarks)):
                        # Se forem valores vazios
                        if not any(video_landmarks[idx_lm]):
                            video_landmarks[idx_lm] = primeira_valida
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
                
                # Geração de Amostras de 20 frames
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
    print(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0])})')
    print(f"\nExtração concluída! Total de {len(features)} amostras coletadas")
    file_.write(f'Shape das features: ({len(features)}, {len(features[0])}, {len(features[0][0])})\n')
    file_.write(f"\nExtração concluída! Total de {len(features)} amostras coletadas\n")
    file_.write(f"\nDetecção de Frames: {frames_detectados}/{frames_totais_global} = {frames_detectados / frames_totais_global * 100}%")
    file_.write(f"\nNúmero de Frames com Padding: {frames_padding_global}/{frames_totais_global+frames_padding_global} = {(frames_padding_global / (frames_totais_global + frames_padding_global) * 100)}%")
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
    if output_path is None:
        output_path = DATASET_TREINO_CSV

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


    for i, (_, df_sample) in enumerate(df.groupby("sample_idx", sort=False)):
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
    DATASET_TREINO_PREPROCESS = "dataset/treino_preprocessed.csv"
    DATASET_TESTE_PREPROCESS = "dataset/teste_preprocessed.csv"


    #print('Gerando dataset de TREINO Preprocessados')
    #extract_features_from_directory(
    #    dataset_root_dir=FRAMES_TREINO_DIR,
    #    sequence_length=SEQUENCE_LENGTH,
    #    step=DEFAULT_STEP,
    #    output_path=DATASET_TREINO_PREPROCESS,
    #    preprocess=True
    #)
#
#
    #print('Gerando dataset de TESTE Preprocessados')
    #extract_features_from_directory(
    #    dataset_root_dir=FRAMES_TESTE_DIR,
    #    sequence_length=SEQUENCE_LENGTH,
    #    step=DEFAULT_STEP,
    #    output_path=DATASET_TESTE_PREPROCESS,
    #    preprocess=True
    #)
    #print('Gerando dataset de TREINO')
    #extract_features_from_directory(
    #    dataset_root_dir=FRAMES_TREINO_DIR,
    #    sequence_length=SEQUENCE_LENGTH,
    #    step=DEFAULT_STEP,
    #    output_path=DATASET_TREINO_CSV,
    #)
#
#
    #print('Gerando dataset de TESTE')
    #extract_features_from_directory(
    #    dataset_root_dir=FRAMES_TESTE_DIR,
    #    sequence_length=SEQUENCE_LENGTH,
    #    step=DEFAULT_STEP,
    #    output_path=DATASET_TESTE_CSV,
    #)
    print('Gerando dataset de TREINO')
    extract_features_from_directory(
        dataset_root_dir=FRAMES_TREINO_DIR,
        sequence_length=SEQUENCE_LENGTH,
        step=DEFAULT_STEP,
        output_path='dataset/treino_15.csv',
        preprocess=True
    )


    print('Gerando dataset de TESTE')
    extract_features_from_directory(
        dataset_root_dir=FRAMES_TESTE_DIR,
        sequence_length=SEQUENCE_LENGTH,
        step=DEFAULT_STEP,
        output_path='dataset/teste_15.csv',
        preprocess=True
    )
