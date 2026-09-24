import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from itertools import product
from time import perf_counter
from utils.constants_cv import HandLandmarker, HandLandmarkerOptions, BaseOptions, VisionRunningMode
from utils.constants import NUM_HANDS, LANDMARKER_PATH, FRAMES_TREINO_DIR, FRAMES_TESTE_DIR



# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def resize_with_padding(image, target_size):
    """
    Redimensiona a imagem mantendo o aspect ratio e adiciona
    padding para atingir exatamente target_size.

    Parameters
    ----------
    image : np.ndarray
        Imagem BGR.
    target_size : tuple
        (width, height)

    Returns
    -------
    np.ndarray
        Imagem redimensionada com padding.
    """

    target_width, target_height = target_size

    original_height, original_width = image.shape[:2]

    # Escala necessária para caber completamente no target
    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # INTER_AREA é apropriado para redução.
    # Para upscale, INTER_CUBIC tende a produzir resultado melhor.
    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # Canvas preto
    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    # Centraliza a imagem
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def apply_clahe(image):
    """
    Aplica CLAHE somente no canal de luminosidade.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpening(image):
    """
    Aplica sharpening moderado.
    """

    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def preprocess_frame(
    frame,
    resolution,
    use_clahe=False,
    use_sharpening=False
):
    """
    Pipeline de pré-processamento.

    Ordem:

        1. Resize + padding
        2. CLAHE
        3. Sharpening
    """
    frame = resize_with_padding(
        frame,
        resolution
    )

    if use_clahe:
        frame = apply_clahe(frame)

    if use_sharpening:
        frame = apply_sharpening(frame)

    return frame

# ============================================================
# MEDIA PIPE
# ============================================================

def process_dataset(
    txt_path,
    resolution,
    use_sharpening,
    use_clahe,
    root_path
):
    """
    Pré-processa cada frame, executa MediaPipe Hands e
    contabiliza a detecção das mãos.

    Um frame é considerado ACERTO se pelo menos uma mão
    for detectada.

    Um frame é considerado FALHA somente se nenhuma mão
    for detectada.

    Parameters
    ----------
    txt_path : str
        Caminho do arquivo TXT onde os resultados serão gravados.

    resolution : tuple
        Exemplo: (1280, 720)

    use_sharpening : bool
        Ativa/desativa sharpening.

    use_clahe : bool
        Ativa/desativa CLAHE.

    root_path : str
        Diretório raiz do dataset.

    Returns
    -------
    dict
        Resumo da execução.
    """

    options = HandLandmarkerOptions(
        base_options=BaseOptions(LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,  # detectar até 2 mãos por frame
    )

    root_path = Path(root_path)
    txt_path = Path(txt_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Diretório raiz não encontrado: {root_path}")

    txt_path.parent.mkdir(parents=True,exist_ok=True)

    total_frames = 0
    frames_at_least_one_detected = 0
    frames_none_detected = 0
    invalid_images = 0
    start_time = perf_counter()

    with HandLandmarker.create_from_options(options) as landmarker:

        for gesture_dir in sorted(root_path.iterdir()):
            for video_dir in sorted(gesture_dir.iterdir()):
                image_files = sorted([p for p in video_dir.iterdir() if p.is_file()])

                for image_path in image_files:
                    frame = cv2.imread(str(image_path))

                    if frame is None:
                        invalid_images += 1
                        continue

                    total_frames += 1
                    frame = preprocess_frame(
                        frame=frame,
                        resolution=resolution,
                        use_clahe=use_clahe,
                        use_sharpening=use_sharpening
                    )

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    results = landmarker.detect(image)

                    if results.hand_landmarks:
                        frames_at_least_one_detected += 1
                    else:
                        frames_none_detected += 1

    elapsed_time = perf_counter() - start_time

    if total_frames > 0:
        success_rate = (frames_at_least_one_detected / total_frames * 100)
        failure_rate = (frames_none_detected / total_frames * 100)


    else:
        success_rate = 0
        failure_rate = 0

    # ========================================================
    # RESULTADO
    # ========================================================

    result = {
        "resolution": (f"{resolution[0]}x{resolution[1]}"),
        "sharpening": use_sharpening,
        "clahe": use_clahe,
        "total_frames": total_frames,
        "frames_at_least_one_hand": (frames_at_least_one_detected),
        "frames_no_hands": (frames_none_detected),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "invalid_images": invalid_images,
        "elapsed_seconds": elapsed_time
    }

    # ========================================================
    # GRAVA RESULTADO
    # ========================================================

    write_header = (
        not txt_path.exists()
        or txt_path.stat().st_size == 0
    )

    with open(txt_path, "a", encoding="utf-8") as file:

        if write_header:
            file.write(
                "resolution;"
                "sharpening;"
                "clahe;"
                "total_frames;"
                "frames_at_least_one_hand;"
                "frames_no_hands;"
                "success_rate;"
                "failure_rate;"
                "invalid_images;"
                "elapsed_seconds\n"
            )

        file.write(
            f"{result['resolution']};"
            f"{result['sharpening']};"
            f"{result['clahe']};"
            f"{result['total_frames']};"
            f"{result['frames_at_least_one_hand']};"
            f"{result['frames_no_hands']};"
            f"{result['success_rate']:.4f};"
            f"{result['failure_rate']:.4f};"
            f"{result['invalid_images']};"
            f"{result['elapsed_seconds']:.2f}\n"
        )

    # ========================================================
    # CONSOLE
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 70)

    print(f"Resolução:       {resolution[0]}x{resolution[1]}")
    print(f"Sharpening:      {use_sharpening}")
    print(f"CLAHE:           {use_clahe}")
    print("-" * 70)
    print(f"Total de frames: {total_frames}")
    print(
        f"Pelo menos 1 mão: "
        f"{frames_at_least_one_detected} "
        f"({success_rate:.2f}%)"
    )
    print(
        f"Nenhuma mão:      "
        f"{frames_none_detected} "
        f"({failure_rate:.2f}%)"
    )
    print(f"Imagens inválidas: {invalid_images}")
    print(f"Tempo: "f"{elapsed_time:.2f} segundos")
    print("=" * 70)

    return result


# ============================================================
# GRID SEARCH
# ============================================================

def run_preprocessing_grid(txt_path, root_path):
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (960, 540),
        (640, 360),
        (480, 270)
    ]

    sharpening_options = [False, True]

    clahe_options = [False, True]

    configurations = list(
        product(
            resolutions,
            sharpening_options,
            clahe_options
        )
    )

    print("\n")
    print("=" * 70)
    print("GRID DE PRÉ-PROCESSAMENTO")
    print("=" * 70)
    print(f"Total de configurações: "f"{len(configurations)}")
    print(f"Frames serão processados novamente "f"para cada configuração.")
    print("=" * 70)

    all_results = []

    for index, (resolution, use_sharpening, use_clahe) in enumerate(configurations, start=1):

        print("\n")
        print(
            f"[{index}/{len(configurations)}] "
            f"Executando configuração..."
        )

        result = process_dataset(
            txt_path=txt_path,
            resolution=resolution,
            use_sharpening=use_sharpening,
            use_clahe=use_clahe,
            root_path=root_path
        )
        all_results.append(result)

    print("\n")
    print("=" * 70)
    print("GRID FINALIZADO")
    print("=" * 70)

    return all_results


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    RESULTS_FILE_TREINO = "logs/resultado_preprocessamento_treino.txt"
    RESULTS_FILE_TESTE = "logs/resultado_preprocessamento_teste.txt"

    results = run_preprocessing_grid(
        txt_path=RESULTS_FILE_TREINO,
        root_path=FRAMES_TREINO_DIR
    )
    results = run_preprocessing_grid(
        txt_path=RESULTS_FILE_TESTE,
        root_path=FRAMES_TESTE_DIR
    )