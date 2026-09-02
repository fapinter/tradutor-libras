import os
import re
import unicodedata

import cv2

from utils.constants import FRAME_RATE, VIDEO_EXTENSIONS


def normalizar_label_video(nome_arquivo_ou_pasta):
    """
    Normaliza o nome do gesto a partir do nome do arquivo ou pasta:
    - Remove extensão (.mp4, etc.)
    - Remove sufixos como '_Articulador3', '_1', etc.
    - Remove acentos ('ç' -> 'c', 'é' -> 'e', etc.)
    - Converte para minúsculas e substitui espaços por hifens.
    Ex: 'Computador portátil_Articulador3.mp4' -> 'computador-portatil'
        'Abraço_Articulador3.mp4' -> 'abraco'
    """
    nome = os.path.splitext(nome_arquivo_ou_pasta)[0]
    nome = re.sub(r"_[Aa]rticulador\w*", "", nome)
    nfkd = unicodedata.normalize("NFKD", nome)
    sem_acento = "".join(
        [c for c in nfkd if not unicodedata.combining(c)])
    limpo = sem_acento.strip().lower()
    limpo = re.sub(r"[\s_]+", "-", limpo)
    return limpo


def extrair_dataset_completo(pasta_videos,
                             pasta_destino_frames):
    """
    Varre a estrutura de pastas e extrai os vídeos automaticamente.
    Suporta:
      1) Estrutura aninhada (subpastas por gesto): pasta_videos/gesto/video.mp4 (ex: videos/treino)
      2) Estrutura plana (vídeos diretamente na pasta): pasta_videos/Gesto_ArticuladorX.mp4 (ex: videos/teste)
    """
    if not os.path.exists(pasta_videos):
        print(
            f"Aviso: A pasta '{pasta_videos}' nao foi encontrada."
        )
        return

    itens = sorted(os.listdir(pasta_videos))
    if not itens:
        print(
            f"Aviso: A pasta '{pasta_videos}' está vazia.")
        return

    subdirs = [
        d for d in itens
        if os.path.isdir(os.path.join(pasta_videos, d))
    ]
    arquivos_video = [
        f for f in itens
        if not os.path.isdir(os.path.join(pasta_videos, f))
        and f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    # Caso 1: Estrutura com subdiretórios por classe (ex: videos/treino/<gesto>/<video>.mp4)
    if subdirs:
        for gesto_dir in subdirs:
            caminho_gesto = os.path.join(pasta_videos,
                                         gesto_dir)
            gesto_label = normalizar_label_video(gesto_dir)
            print(
                f"\nProcessando vídeos do gesto: {gesto_label}..."
            )

            for nome_video in sorted(
                    os.listdir(caminho_gesto)):
                if nome_video.lower().endswith(
                        VIDEO_EXTENSIONS):
                    caminho_video = os.path.join(
                        caminho_gesto, nome_video)
                    print(f" -> Extraindo: {nome_video}")
                    extract_frames(
                        video_path=caminho_video,
                        output_root_dir=pasta_destino_frames,
                        gesture_label=gesto_label,
                    )

    # Caso 2: Estrutura plana com vídeos diretos na pasta (ex: videos/teste/Abacaxi_Articulador3.mp4)
    if arquivos_video:
        print(
            f"\nProcessando {len(arquivos_video)} vídeo(s) na pasta '{pasta_videos}'..."
        )
        for nome_video in arquivos_video:
            caminho_video = os.path.join(pasta_videos,
                                         nome_video)
            gesto_label = normalizar_label_video(nome_video)
            print(
                f" -> Extraindo: {nome_video} (classe: '{gesto_label}')"
            )
            extract_frames(
                video_path=caminho_video,
                output_root_dir=pasta_destino_frames,
                gesture_label=gesto_label,
            )


def extract_frames(video_path,
                   output_root_dir,
                   gesture_label,
                   frame_rate=FRAME_RATE):
    """
    Extrai frames de um vídeo e os salva em uma subpasta exclusiva desse vídeo.

    Estrutura gerada:
        output_root_dir/gesture_label/v0000/frame_0.jpg
        output_root_dir/gesture_label/v0001/frame_0.jpg  ← próximo vídeo do mesmo gesto

    Cada chamada cria uma nova subpasta v{N:04d}, nunca sobrescreve vídeos anteriores.
    O parâmetro frame_rate define quantos frames pular entre capturas (não é fps):
        frame_rate=5 captura 1 frame a cada 5 → ~6 fps para vídeos a 30fps.
    """
    gesture_dir = os.path.join(output_root_dir,
                               gesture_label)
    os.makedirs(gesture_dir, exist_ok=True)

    # Conta subpastas de vídeo já existentes para gerar nome único
    existing_video_dirs = [
        d for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d))
        and d.startswith("v")
    ]
    video_idx = len(existing_video_dirs)
    output_dir = os.path.join(gesture_dir,
                              f"v{video_idx:04d}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_path = os.path.join(
                output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        count += 1

    cap.release()
    print(
        f"Frames da classe '{gesture_label}' salvos em {output_dir} ({frame_count} frames)"
    )
