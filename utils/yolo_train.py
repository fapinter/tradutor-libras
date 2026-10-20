"""
Treinamento do YOLO-Pose para detecção dos 21 landmarks das mãos (dataset hand-keypoints da Ultralytics).

Executar a partir da raiz do projeto, como módulo:
    python -m utils.yolo_train

Rodar como `python utils/yolo_train.py` coloca utils/ no sys.path, e os arquivos utils/tqdm.py e
utils/xgboost.py passam a sobrescrever as bibliotecas de mesmo nome usadas pela Ultralytics.

O dataset (369 MB) é baixado automaticamente no primeiro treino, no diretório datasets_dir das
configurações da Ultralytics (`yolo settings`). Ao final, o melhor modelo é copiado para
YOLO_HAND_MODEL_PATH, caminho utilizado pelo tests_extraction.py.
"""
import os
import shutil

import torch
from ultralytics import YOLO

from utils.constants import (
    SEED,
    YOLO_DATASET_YAML,
    YOLO_HAND_MODEL_PATH,
    YOLO_MODELO_BASE,
)

YOLO_RUNS_DIR = "runs/pose"
YOLO_RUN_NAME = "yolo_hand_pose"


def verificar_gpu():
    """Garante que o treinamento será feito na GPU, interrompendo a execução caso não haja CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não disponível: o treinamento não será executado na CPU.\n"
            f"PyTorch {torch.__version__} (CUDA: {torch.version.cuda}).\n"
            "Verifique o driver da NVIDIA (nvidia-smi) e instale o PyTorch com suporte a CUDA, ex.:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB) | "
          f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    return 0


def treinar_yolo(
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    workers=8,
    resume=False
):
    """
    Treina o YOLO-Pose a partir de YOLO_MODELO_BASE e copia o melhor modelo para YOLO_HAND_MODEL_PATH.

    resume=True continua um treinamento interrompido a partir do último checkpoint (last.pt).
    """
    device = verificar_gpu()
    run_dir = os.path.join(YOLO_RUNS_DIR, YOLO_RUN_NAME)

    if resume:
        last_path = os.path.join(run_dir, "weights", "last.pt")
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"Checkpoint '{last_path}' não encontrado para continuar o treino.")
        print(f"Continuando treinamento a partir de {last_path}")
        model = YOLO(last_path)
        model.train(resume=True, device=device)
    else:
        if not os.path.exists(YOLO_DATASET_YAML):
            raise FileNotFoundError(f"YAML do dataset '{YOLO_DATASET_YAML}' não encontrado.")
        model = YOLO(YOLO_MODELO_BASE)
        model.train(
            data=YOLO_DATASET_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            workers=workers,
            device=device,
            amp=True,  # precisão mista (FP16), acelera o treino na GPU
            seed=SEED,
            project=YOLO_RUNS_DIR,
            name=YOLO_RUN_NAME,
            exist_ok=True,  # nome fixo para permitir o resume (sobrescreve um treino anterior)
            plots=True,
        )

    best_path = model.trainer.best
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Melhor modelo '{best_path}' não encontrado após o treinamento.")

    # Avaliação final do melhor modelo no conjunto de validação
    metrics = YOLO(best_path).val(data=YOLO_DATASET_YAML, imgsz=imgsz, batch=batch, device=device)
    print(f"Box  mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
    print(f"Pose mAP50: {metrics.pose.map50:.4f} | mAP50-95: {metrics.pose.map:.4f}")

    os.makedirs(os.path.dirname(YOLO_HAND_MODEL_PATH), exist_ok=True)
    shutil.copy(best_path, YOLO_HAND_MODEL_PATH)
    print(f"Modelo copiado: {best_path} -> {YOLO_HAND_MODEL_PATH}")


if __name__ == "__main__":
    treinar_yolo(
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        workers=8,
        resume=False
    )
