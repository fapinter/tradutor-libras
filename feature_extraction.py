import os
import re

import mediapipe as mp
import numpy as np
import pandas as pd

from utils import NUM_FEATURES, extrair_ambas_maos

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_N_MAO = 63  # features de uma mão
_ZEROS_MAO = [0.0] * _N_MAO  # placeholder para mão ausente


def _extrair_numero_frame(filename):
    matches = re.findall(r"\d+", filename)
    return int(matches[0]) if matches else 0


def _listar_frames(directory):
    """Retorna lista de imagens ordenada numericamente de forma robusta."""
    file_names = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    file_names.sort(key=_extrair_numero_frame)
    return file_names


def _video_dirs_de_classe(class_dir):
    """
    Retorna os diretórios de vídeo para uma classe.
    Subpastas por vídeo: class_dir/v0000/, class_dir/v0001/, ...
    """
    subdirs = sorted(
        [
            os.path.join(class_dir, d)
            for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ]
    )
    return subdirs if subdirs else [class_dir]


def extract_features_from_directory(
    dataset_root_dir,
    model_asset_path="hand_landmarker.task",
    mode="lstm",
    sequence_length=20,
    step=1,
    export_dataframe=False,
    output_csv_path=None,
):
    """
    Varre os diretórios de imagens e extrai sequências 3D de coordenadas normalizadas de AMBAS as mãos para o LSTM.

    Retorna features 3D no formato: (amostras, frames, 126_features).
    """
    if not os.path.exists(dataset_root_dir):
        print(f"[ERRO] Diretório '{dataset_root_dir}' não foi encontrado.")
        return [], []

    features = []
    labels = []

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,  # detectar até 2 mãos por frame
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_name in sorted(os.listdir(dataset_root_dir)):
            class_dir = os.path.join(dataset_root_dir, label_name)

            if not os.path.isdir(class_dir):
                continue

            print(f"Extraindo features do gesto: {label_name}...")

            video_dirs = _video_dirs_de_classe(class_dir)
            sequencias_classe = 0

            for video_dir in video_dirs:
                file_names = _listar_frames(video_dir)
                if not file_names:
                    continue

                video_landmarks = []
                ultimo_valido = [0.0] * NUM_FEATURES

                for file_name in file_names:
                    image_path = os.path.join(video_dir, file_name)
                    try:
                        mp_image = mp.Image.create_from_file(image_path)
                        results = landmarker.detect(mp_image)

                        if results.hand_landmarks:
                            landmarks = extrair_ambas_maos(
                                results.hand_landmarks,
                                results.handedness,
                            )
                            ultimo_valido = landmarks
                            video_landmarks.append(landmarks)
                        else:
                            # Forward-fill com o último frame válido
                            video_landmarks.append(ultimo_valido)

                    except Exception as e:
                        print(f"Erro ao processar {image_path}: {e}")

                # Back-fill em frames iniciais sem detecção utilizando a primeira mão válida encontrada
                primeira_valida = next((lm for lm in video_landmarks if any(lm)), None)
                if primeira_valida is not None:
                    for idx_lm in range(len(video_landmarks)):
                        if not any(video_landmarks[idx_lm]):
                            video_landmarks[idx_lm] = primeira_valida
                        else:
                            break

                for i in range(0, len(video_landmarks) - sequence_length + 1, step):
                    features.append(video_landmarks[i : i + sequence_length])
                    labels.append(label_name)
                    sequencias_classe += 1

            if sequencias_classe == 0:
                print(
                    f"  AVISO: '{label_name}' gerou 0 sequências. "
                    f"Verifique se os vídeos têm >= {sequence_length} frames detectáveis."
                )
            else:
                print(f"  -> {sequencias_classe} sequência(s) para '{label_name}'")

    print(
        f"\nExtração concluída! Total (LSTM): {len(features)} amostras, {NUM_FEATURES} features/frame"
    )

    if export_dataframe:
        _exportar_csv(features, labels, output_path=output_csv_path)

    return features, labels


def _exportar_csv(features, labels, output_path=None):
    """Salva o dataset em CSV para LSTM."""
    coord_cols = []
    for prefixo in ("d", "e"):  # d = direita, e = esquerda
        for i in range(1, 22):
            coord_cols += [f"{prefixo}_x_{i}", f"{prefixo}_y_{i}", f"{prefixo}_z_{i}"]

    if output_path is None:
        output_path = "./dataset/dataset_completo_lstm.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exportando {len(labels)} amostras...")
    rows = []
    for sample_idx, (seq, label) in enumerate(zip(features, labels)):
        for frame_idx, frame in enumerate(seq):
            row = [label, sample_idx, frame_idx] + list(frame)
            rows.append(row)
    all_cols = ["target", "sample_idx", "frame_idx"] + coord_cols
    df = pd.DataFrame(rows, columns=all_cols)

    df.to_csv(output_path, index=False)
    print(f"Dataset exportado: {output_path}")


def import_from_csv(filepath: str, mode: str = "lstm"):
    """
    Carrega o dataset a partir de um CSV no formato 3D para o modelo LSTM: (amostras, frames, features).
    """
    df = pd.read_csv(filepath)

    feature_cols = [
        col for col in df.columns if col not in ["target", "frame_idx", "sample_idx"]
    ]
    unique_samples = df["sample_idx"].unique()
    num_samples = len(unique_samples)
    num_frames = df["frame_idx"].nunique()
    num_features = len(feature_cols)

    features = np.zeros((num_samples, num_frames, num_features))
    labels = []

    for i, (sample_id, df_sample) in enumerate(df.groupby("sample_idx", sort=False)):
        df_sample = df_sample.sort_values(by="frame_idx")

        landmarks_matrix = df_sample[feature_cols].values
        t_real = landmarks_matrix.shape[0]
        features[i, :t_real, :] = landmarks_matrix

        sample_label = df_sample["target"].iloc[0]
        labels.append(sample_label)

    labels = np.array(labels)
    return features, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extração de Features de Gestos de Libras (LSTM)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/frames_treino",
        help="Caminho para o diretório raiz dos frames (ex: dataset/frames_treino ou dataset/frames_teste)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho do CSV de saída (ex: dataset/dataset_completo_lstm.csv ou dataset/dataset_teste_lstm.csv)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=20,
        help="Tamanho da janela temporal de frames (padrão: 20)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Passo da janela deslizante (padrão: 1)",
    )
    parser.add_argument(
        "--no_export",
        action="store_true",
        help="Se definido, não exporta para CSV",
    )

    args = parser.parse_args()

    print(f"=== Extração de Features para LSTM ({args.dataset}) ===")
    extract_features_from_directory(
        dataset_root_dir=args.dataset,
        mode="lstm",
        sequence_length=args.sequence_length,
        step=args.step,
        export_dataframe=not args.no_export,
        output_csv_path=args.output,
    )
