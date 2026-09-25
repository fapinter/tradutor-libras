import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from itertools import product
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from utils.constants_cv import (
    HandLandmarker,
    HandLandmarkerOptions,
    BaseOptions,
    VisionRunningMode
)
from utils.constants import (
    NUM_HANDS,
    LANDMARKER_PATH,
    FRAMES_TREINO_DIR,
    FRAMES_TESTE_DIR
)


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
    """

    target_width, target_height = target_size
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

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

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

    return cv2.filter2D(image, -1, kernel)

def apply_blur(image, blur_type=None, kernel_size=3):
    if blur_type == None:
        return image

    match(blur_type):
        case "median":
            return cv2.medianBlur(image,kernel_size)
        case "gaussian":
            return cv2.medianBlur(image, kernel_size)
        case "bilateral":
            return cv2.bilateralFilter(image,d=kernel_size,sigmaColor=75,sigmaSpace=75)


def preprocess_frame(
    frame,
    resolution,
    use_clahe=False,
    use_sharpening=False,
    blur_type=None,
    blur_kernel_size=3
):
    """
    Pipeline:

        1. Resize + padding
        2. CLAHE
        3. Sharpening
    """

    frame = resize_with_padding(
        frame,
        resolution
    )

    frame = apply_blur(
        frame,
        blur_type=blur_type,
        kernel_size=blur_kernel_size
    )

    if use_clahe:
        frame = apply_clahe(frame)

    if use_sharpening:
        frame = apply_sharpening(frame)

    return frame


# ============================================================
# AUXILIARES DE PARALELISMO
# ============================================================

def configure_worker():
    """
    Configuração executada uma vez em cada processo worker.

    O OpenCV pode criar threads internamente. Como estamos usando
    multiprocessing, limitar essas threads evita oversubscription
    (vários processos × várias threads por processo).
    """

    cv2.setNumThreads(1)

    # Evita que algumas bibliotecas matemáticas criem várias
    # threads dentro de cada processo.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def get_image_files(root_path):
    """
    Retorna todos os arquivos de imagem seguindo:

        root/
            gesto/
                video/
                    imagem

    A lista é criada uma única vez por processo.
    """

    root_path = Path(root_path)

    image_files = []

    for gesture_dir in root_path.iterdir():

        if not gesture_dir.is_dir():
            continue

        for video_dir in gesture_dir.iterdir():

            if not video_dir.is_dir():
                continue

            for image_path in video_dir.iterdir():

                if image_path.is_file():
                    image_files.append(image_path)
    return image_files


# ============================================================
# PROCESSAMENTO DE UMA CONFIGURAÇÃO
# ============================================================

def process_dataset(
    txt_path,
    resolution,
    use_sharpening,
    use_clahe,
    blur_type,
    blur_kernel_size,
    root_path
):
    """
    Executa uma configuração completa.

    Esta função é mantida como API pública para executar uma
    configuração individual.

    A gravação do resultado é feita somente pelo processo
    principal quando chamada pela grid paralela.
    """

    result = _process_dataset_core(
        resolution=resolution,
        use_sharpening=use_sharpening,
        use_clahe=use_clahe,
        blur_type=blur_type,
        blur_kernel_size=blur_kernel_size,
        root_path=root_path
    )

    append_result_to_file(
        txt_path,
        result
    )

    print_result(result)
    return result


def _process_dataset_core(
    resolution,
    use_sharpening,
    use_clahe,
    blur_type,
    blur_kernel_size,
    root_path
):
    """
    Núcleo do processamento.

    IMPORTANTE:
    Esta função não escreve no arquivo de resultados.

    Isso é necessário porque vários processos podem terminar
    simultaneamente. Somente o processo principal deve escrever
    no TXT.
    """

    root_path = Path(root_path)

    if not root_path.exists():
        raise FileNotFoundError(
            f"Diretório raiz não encontrado: {root_path}"
        )

    total_frames = 0
    frames_at_least_one_detected = 0
    frames_none_detected = 0

    start_time = perf_counter()

    # Cada processo cria sua própria instância do MediaPipe.
    options = HandLandmarkerOptions(
        base_options=BaseOptions(LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS
    )

    # O processo cria sua própria lista de imagens.
    image_files = get_image_files(root_path)

    with HandLandmarker.create_from_options(options) as landmarker:

        for image_path in image_files:

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            total_frames += 1

            frame = preprocess_frame(
                frame=frame,
                resolution=resolution,
                use_clahe=use_clahe,
                use_sharpening=use_sharpening,
                blur_type=blur_type,
                blur_kernel_size=blur_kernel_size
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            results = landmarker.detect(image)

            if results.hand_landmarks:
                frames_at_least_one_detected += 1
            else:
                frames_none_detected += 1

    elapsed_time = perf_counter() - start_time

    if total_frames > 0:

        success_rate = (
            frames_at_least_one_detected
            / total_frames
            * 100
        )

        failure_rate = (
            frames_none_detected
            / total_frames
            * 100
        )

    else:

        success_rate = 0
        failure_rate = 0

    return {
        "resolution": f"{resolution[0]}x{resolution[1]}",
        "sharpening": use_sharpening,
        "clahe": use_clahe,
        "blur_type": blur_type,
        "blur_kernel_size": blur_kernel_size,
        "total_frames": total_frames,
        "frames_at_least_one_hand": (
            frames_at_least_one_detected
        ),
        "frames_no_hands": (
            frames_none_detected
        ),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "elapsed_seconds": elapsed_time
    }


# ============================================================
# RESULTADOS
# ============================================================

def append_result_to_file(txt_path, result):
    """
    O arquivo é escrito somente pelo processo principal.
    """

    txt_path = Path(txt_path)

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_header = (
        not txt_path.exists()
        or txt_path.stat().st_size == 0
    )

    with open(
        txt_path,
        "a",
        encoding="utf-8"
    ) as file:

        if write_header:
            file.write(
                "resolution;"
                "sharpening;"
                "clahe;"
                "blur_type;"
                "blur_kernel_size;"
                "total_frames;"
                "frames_at_least_one_hand;"
                "frames_no_hands;"
                "success_rate;"
                "failure_rate;"
                "elapsed_seconds\n"
            )

        file.write(
            f"{result['resolution']};"
            f"{result['sharpening']};"
            f"{result['clahe']};"
            f"{result['blur_type']};"
            f"{result['blur_kernel_size']};"
            f"{result['total_frames']};"
            f"{result['frames_at_least_one_hand']};"
            f"{result['frames_no_hands']};"
            f"{result['success_rate']:.4f};"
            f"{result['failure_rate']:.4f};"
            f"{result['elapsed_seconds']:.2f}\n"
        )


def print_result(result):

    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO FINALIZADA")
    print("=" * 70)

    print(
        f"Resolução:        {result['resolution']}"
    )

    print(
        f"Sharpening:       {result['sharpening']}"
    )

    print(
        f"CLAHE:            {result['clahe']}"
    )
    print(
        f"Blur Type:        {result['blur_type']}"
    )
    print(
        f"Blur Kernel Size: {result['blur_kernel_size']}"
    )

    print("-" * 70)

    print(
        f"Total de frames:  "
        f"{result['total_frames']}"
    )

    print(
        f"Pelo menos 1 mão: "
        f"{result['frames_at_least_one_hand']} "
        f"({result['success_rate']:.2f}%)"
    )

    print(
        f"Nenhuma mão:      "
        f"{result['frames_no_hands']} "
        f"({result['failure_rate']:.2f}%)"
    )

    print(
        f"Tempo:            "
        f"{result['elapsed_seconds']:.2f} segundos"
    )

    print("=" * 70)


# ============================================================
# WORKER DA GRID
# ============================================================

def _run_configuration_worker(args):
    """
    Worker executado em um processo separado.

    O argumento é uma tupla para facilitar o uso com
    ProcessPoolExecutor.
    """

    (
        index,
        total_configurations,
        resolution,
        use_sharpening,
        use_clahe,
        blur_type,
        blur_kernel_size,
        root_path
    ) = args

    configure_worker()

    print(
        f"[WORKER {os.getpid()}] "
        f"[{index}/{total_configurations}] "
        f"{resolution[0]}x{resolution[1]} | "
        f"sharpening={use_sharpening} | "
        f"clahe={use_clahe}",
        flush=True
    )

    start_time = perf_counter()

    result = _process_dataset_core(
        resolution=resolution,
        use_sharpening=use_sharpening,
        use_clahe=use_clahe,
        root_path=root_path,
        blur_type=blur_type,
        blur_kernel_size=blur_kernel_size
    )

    result["worker_pid"] = os.getpid()

    # O tempo já é calculado no core.
    # Mantemos também o tempo externo para debug.
    result["worker_elapsed_seconds"] = (
        perf_counter() - start_time
    )

    return index, result


# ============================================================
# GRID SEARCH PARALELA
# ============================================================

def run_preprocessing_grid(
    txt_path,
    root_path,
    max_workers=None
):
    """
    Executa todas as combinações em paralelo.

    5 resoluções × 2 sharpening × 2 CLAHE = 20 configurações.

    Parameters
    ----------
    txt_path : str
        Arquivo TXT dos resultados.

    root_path : str
        Dataset.

    max_workers : int | None
        Número de processos simultâneos.

        None:
            usa a quantidade padrão do ProcessPoolExecutor.

        Recomenda-se começar com 2 ou 4 para MediaPipe e
        aumentar somente se CPU/RAM permitirem.
    """

    resolutions = [
        (1920, 1080),
        (1280, 720),
        (960, 540),
        (640, 360),
        (480, 270)
    ]

    sharpening_options = [False]
    clahe_options = [False]
    blur_type_options = ["median", "gaussian", "bilateral"]
    blur_kernel_options = [3,5]

    configurations = list(
        product(
            resolutions,
            sharpening_options,
            clahe_options,
            blur_type_options,
            blur_kernel_options
        )
    )

    total_configurations = len(configurations)

    print("\n")
    print("=" * 70)
    print("GRID DE PRÉ-PROCESSAMENTO PARALELA")
    print("=" * 70)

    print(
        f"Total de configurações: "
        f"{total_configurations}"
    )

    print(
        f"Processos simultâneos: "
        f"{max_workers if max_workers is not None else 'padrão'}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Importante:
    # Não apagamos automaticamente o arquivo.
    #
    # Se quiser começar um experimento do zero, apague o TXT
    # antes da execução.
    # --------------------------------------------------------

    tasks = [
        (
            index,
            total_configurations,
            resolution,
            use_sharpening,
            use_clahe,
            blur_type,
            blur_kernel_size,
            str(root_path)
        )
        for index, (resolution,use_sharpening,use_clahe, blur_type, blur_kernel_size)
        in enumerate(configurations, start=1)
    ]

    all_results = []

    grid_start = perf_counter()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=configure_worker
    ) as executor:

        futures = [
            executor.submit(
                _run_configuration_worker,
                task
            )
            for task in tasks
        ]

        for completed, future in enumerate(
            as_completed(futures),
            start=1
        ):

            try:

                index, result = future.result()

                # Somente o processo principal escreve no arquivo.
                append_result_to_file(
                    txt_path,
                    result
                )

                all_results.append(result)

                print(
                    f"\n[{completed}/{total_configurations}] "
                    f"Configuração {index} concluída.",
                    flush=True
                )

                print_result(result)

            except Exception as exc:

                print(
                    f"\nERRO em uma configuração: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

    total_elapsed = perf_counter() - grid_start

    print("\n")
    print("=" * 70)
    print("GRID FINALIZADO")
    print("=" * 70)

    print(
        f"Configurações concluídas: "
        f"{len(all_results)}/{total_configurations}"
    )

    print(
        f"Tempo total: "
        f"{total_elapsed:.2f} segundos "
        f"({total_elapsed / 60:.2f} minutos)"
    )

    print("=" * 70)

    return all_results

if __name__ == "__main__":
    RESULTS_FILE_TREINO = "logs/resultado_preprocessamento_treino_blur.txt"
    RESULTS_FILE_TESTE ="logs/resultado_preprocessamento_teste_blur.txt"

    # Número de combinações rodando em paralelo
    MAX_WORKERS = 4

    results_treino = run_preprocessing_grid(
        txt_path=RESULTS_FILE_TREINO,
        root_path=FRAMES_TREINO_DIR,
        max_workers=MAX_WORKERS
    )

    results_teste = run_preprocessing_grid(
        txt_path=RESULTS_FILE_TESTE,
        root_path=FRAMES_TESTE_DIR,
        max_workers=MAX_WORKERS
    )
