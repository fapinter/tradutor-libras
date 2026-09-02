import os
import re

import mediapipe as mp
import numpy as np
import pandas as pd

from utils.constants import (
    DATASET_TREINO_CSV,
    DEFAULT_STEP,
    FRAMES_TREINO_DIR,
    IMAGE_EXTENSIONS,
    LANDMARKER_PATH,
    NUM_FEATURES,
    NUM_HANDS,
    SEQUENCE_LENGTH,
)
from utils.constants_cv import (
    BaseOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    VisionRunningMode,
)
from utils.utils import extrair_ambas_maos


def _extrair_numero_frame(filename):
    matches = re.findall(r"\d+", filename)
    return int(matches[0]) if matches else 0


def _listar_frames(directory):
    """Retorna lista de imagens ordenada numericamente de forma robusta."""
    file_names = [
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    file_names.sort(key=_extrair_numero_frame)
    return file_names


def _video_dirs_de_classe(class_dir):
    """
    Retorna os diretórios de vídeo para uma classe.
    Subpastas por vídeo: class_dir/v0000/, class_dir/v0001/, ...
    """
    subdirs = sorted([
        os.path.join(class_dir, d)
        for d in os.listdir(class_dir)
        if os.path.isdir(os.path.join(class_dir, d))
    ])
    return subdirs if subdirs else [class_dir]


def extract_features_from_directory(
    dataset_root_dir,
    LANDMARKER_PATH=LANDMARKER_PATH,
    mode="lstm",
    sequence_length=SEQUENCE_LENGTH,
    step=DEFAULT_STEP,
    export_dataframe=False,
    output_csv_path=None,
    return_groups=False,
):
    """
    Varre os diretórios de imagens e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos para o LSTM.

    Retorna features 3D no formato: (amostras, frames, 126_features).
    """
    if not os.path.exists(dataset_root_dir):
        print(
            f"[ERRO] Diretório '{dataset_root_dir}' não foi encontrado."
        )
        return ([], [], []) if return_groups else ([], [])

    features = []
    labels = []
    groups = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            LANDMARKER_PATH=LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,  # detectar até 2 mãos por frame
    )

    with HandLandmarker.create_from_options(
            options) as landmarker:
        for label_name in sorted(
                os.listdir(dataset_root_dir)):
            class_dir = os.path.join(dataset_root_dir,
                                     label_name)

            if not os.path.isdir(class_dir):
                continue

            print(
                f"Extraindo features do gesto: {label_name}..."
            )

            video_dirs = _video_dirs_de_classe(class_dir)
            sequencias_classe = 0

            for video_dir in video_dirs:
                file_names = _listar_frames(video_dir)
                if not file_names:
                    continue

                video_id = f"{label_name}_{os.path.basename(video_dir)}"
                video_landmarks = []
                ultimo_valido = [0.0] * NUM_FEATURES

                for file_name in file_names:
                    image_path = os.path.join(
                        video_dir, file_name)
                    try:
                        mp_image = mp.Image.create_from_file(
                            image_path)
                        results = landmarker.detect(
                            mp_image)

                        if results.hand_landmarks:
                            landmarks = extrair_ambas_maos(
                                results.hand_landmarks,
                                results.handedness,
                            )
                            ultimo_valido = landmarks
                            video_landmarks.append(
                                landmarks)
                        else:
                            # Forward-fill com o último frame válido
                            video_landmarks.append(
                                ultimo_valido)

                    except Exception as e:
                        print(
                            f"Erro ao processar {image_path}: {e}"
                        )

                # Back-fill em frames iniciais sem detecção utilizando a primeira mão válida encontrada
                primeira_valida = next(
                    (lm
                     for lm in video_landmarks if any(lm)),
                    None)
                if primeira_valida is not None:
                    for idx_lm in range(
                            len(video_landmarks)):
                        if not any(video_landmarks[idx_lm]):
                            video_landmarks[
                                idx_lm] = primeira_valida
                        else:
                            break

                for i in range(
                        0,
                        len(video_landmarks) -
                        sequence_length + 1, step):
                    features.append(
                        video_landmarks[i:i +
                                        sequence_length])
                    labels.append(label_name)
                    groups.append(video_id)
                    sequencias_classe += 1

            if sequencias_classe == 0:
                print(
                    f"  AVISO: '{label_name}' gerou 0 sequências. "
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis."
                )
            else:
                print(
                    f"  -> {sequencias_classe} sequência(s) para '{label_name}'"
                )

    print(
        f"\nExtração concluída! Total (LSTM): {len(features)} amostras, {NUM_FEATURES} features/frame"
    )

    if export_dataframe:
        _exportar_csv(features,
                      labels,
                      groups=groups,
                      output_path=output_csv_path)

    if return_groups:
        return features, labels, np.array(groups)
    return features, labels


def _exportar_csv(features,
                  labels,
                  groups=None,
                  output_path=None):
    """Salva o dataset em CSV para LSTM, incluindo video_id para agrupamento."""
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
    tem_grupos = groups is not None and len(groups) == len(
        labels)

    for sample_idx, (seq, label) in enumerate(
            zip(features, labels)):
        group_val = groups[
            sample_idx] if tem_grupos else f"sample_{sample_idx}"
        for frame_idx, frame in enumerate(seq):
            row = [label, group_val, sample_idx, frame_idx
                   ] + list(frame)
            rows.append(row)

    all_cols = [
        "target", "video_id", "sample_idx", "frame_idx"
    ] + coord_cols
    df = pd.DataFrame(rows, columns=all_cols)

    df.to_csv(output_path, index=False)
    print(f"Dataset exportado: {output_path}")


def import_from_csv(filepath: str,
                    mode: str = "lstm",
                    return_groups: bool = False):
    """
    Carrega o dataset a partir de um CSV no formato 3D para o modelo LSTM: (amostras, frames, features).
    Se return_groups=True, retorna também o array de identificadores de vídeo/grupo (para StratifiedGroupKFold).
    """
    df = pd.read_csv(filepath)

    feature_cols = [
        col for col in df.columns if col not in
        ["target", "video_id", "frame_idx", "sample_idx"]
    ]
    unique_samples = df["sample_idx"].unique()
    num_samples = len(unique_samples)
    num_frames = df["frame_idx"].nunique()
    num_features = len(feature_cols)

    features = np.zeros(
        (num_samples, num_frames, num_features))
    labels = []
    groups = []

    tem_coluna_video = "video_id" in df.columns

    for i, (sample_id, df_sample) in enumerate(
            df.groupby("sample_idx", sort=False)):
        df_sample = df_sample.sort_values(by="frame_idx")

        landmarks_matrix = df_sample[feature_cols].values
        t_real = landmarks_matrix.shape[0]
        features[i, :t_real, :] = landmarks_matrix

        sample_label = df_sample["target"].iloc[0]
        labels.append(sample_label)

        if tem_coluna_video:
            groups.append(df_sample["video_id"].iloc[0])
        else:
            groups.append(str(sample_id))

    labels = np.array(labels)
    groups = np.array(groups)

    if return_groups:
        return features, labels, groups
    return features, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Extração de Features de Gestos de Libras (LSTM)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=FRAMES_TREINO_DIR,
        help=
        "Caminho para o diretório raiz dos frames (ex: dataset/frames_treino ou dataset/frames_teste)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=
        "Caminho do CSV de saída (ex: dataset/dataset_completo_lstm.csv ou dataset/dataset_teste_lstm.csv)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=
        f"Tamanho da janela temporal de frames (padrão: {SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=
        f"Passo da janela deslizante (padrão: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--no_export",
        action="store_true",
        help="Se definido, não exporta para CSV",
    )

    args = parser.parse_args()

    print(
        f"=== Extração de Features para LSTM ({args.dataset}) ==="
    )
    extract_features_from_directory(
        dataset_root_dir=args.dataset,
        mode="lstm",
        sequence_length=args.sequence_length,
        step=args.step,
        export_dataframe=not args.no_export,
        output_csv_path=args.output,
    )
