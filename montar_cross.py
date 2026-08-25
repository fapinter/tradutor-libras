from pathlib import Path
from collections import defaultdict
import csv
import re
import shutil
import unicodedata

BASE = Path(__file__).resolve().parent

MINDS_VIDEOS = BASE / "MINDS" / "videos"
MALTA_MANIFEST = BASE / "MALTA_MINDS" / "manifest_malta.csv"

OUTPUT = BASE / "MINDS_MALTA_CROSS"

# VIDEOS_PER_DATASET = 2


def normalize(text):
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", "", text)


# ======================================================
# LIMPAR PASTA DE SAÍDA
# ======================================================

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir(parents=True)


# ======================================================
# LER MALTA
# ======================================================

malta_by_label = defaultdict(list)

with open(
    MALTA_MANIFEST,
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if row["status"] not in ("downloaded", "existing"):
            continue

        label = normalize(row["label"])

        file_path = BASE / row["file"]

        if not file_path.exists():
            continue

        malta_by_label[label].append({
            "path": file_path,
            "source": row["source"],
        })


classes = sorted(malta_by_label.keys())

print(f"Classes encontradas no MALTA: {len(classes)}")


# ======================================================
# LOCALIZAR VÍDEOS MINDS
# ======================================================

minds_files = list(MINDS_VIDEOS.glob("*.mp4"))

print(f"Vídeos encontrados no MINDS: {len(minds_files)}")


def find_minds_videos(label):

    matches = []

    for video in minds_files:

        normalized_name = normalize(video.stem)

        if label in normalized_name:

            signer_match = re.search(
                r"sinalizador(\d+)",
                normalized_name
            )

            signer = (
                signer_match.group(1)
                if signer_match
                else video.name
            )

            matches.append({
                "path": video,
                "signer": signer,
            })

    return matches


# ======================================================
# SELECIONAR DIVERSIDADE
# ======================================================

def select_different(items, key, amount):

    selected = []
    used = set()

    # Primeiro tenta pegar diferentes fontes/sinalizantes
    for item in items:

        value = item[key]

        if value in used:
            continue

        selected.append(item)
        used.add(value)

        if len(selected) == amount:
            return selected

    # Se não houver diversidade suficiente,
    # completa com os restantes
    for item in items:

        if item in selected:
            continue

        selected.append(item)

        if len(selected) == amount:
            break

    return selected


# ======================================================
# MONTAGEM
# ======================================================

metadata = []

for label in classes:

    print(f"\n=== {label.upper()} ===")

    # ----------------------
    # MALTA
    # ----------------------

    malta_candidates = malta_by_label[label]
    '''
    malta_selected = select_different(
        malta_candidates,
        "source",
        VIDEOS_PER_DATASET
    )
    '''

    
    malta_selected = malta_candidates

    # ----------------------
    # MINDS
    # ----------------------


    minds_candidates = find_minds_videos(label)
    '''
    minds_selected = select_different(
        minds_candidates,
        "signer",
        VIDEOS_PER_DATASET
    )
    '''

    minds_selected = minds_candidates
    
    print(
        f"MINDS: {len(minds_candidates)} disponíveis"
    )

    print(
        f"MALTA: {len(malta_candidates)} disponíveis"
    )
    '''

    if len(minds_selected) < VIDEOS_PER_DATASET:

        print(
            f"ERRO: poucos vídeos MINDS para {label}"
        )

        continue

    if len(malta_selected) < VIDEOS_PER_DATASET:

        print(
            f"ERRO: poucos vídeos MALTA para {label}"
        )

        continue
    '''
    # ----------------------
    # DIRETÓRIOS
    # ----------------------

    class_dir = OUTPUT / label

    minds_dir = class_dir / "minds"
    malta_dir = class_dir / "malta"

    minds_dir.mkdir(parents=True)
    malta_dir.mkdir(parents=True)

    # ----------------------
    # COPIAR MINDS
    # ----------------------

    for index, item in enumerate(
        minds_selected,
        start=1
    ):

        destination = (
            minds_dir /
            f"minds_{index:02d}.mp4"
        )

        shutil.copy2(
            item["path"],
            destination
        )

        metadata.append({
            "label": label,
            "dataset": "MINDS",
            "origin": f"Sinalizador{item['signer']}",
            "file": str(
                destination.relative_to(OUTPUT)
            )
        })

    # ----------------------
    # COPIAR MALTA
    # ----------------------

    for index, item in enumerate(
        malta_selected,
        start=1
    ):

        destination = (
            malta_dir /
            f"malta_{index:02d}.mp4"
        )

        shutil.copy2(
            item["path"],
            destination
        )

        metadata.append({
            "label": label,
            "dataset": "MALTA",
            "origin": item["source"],
            "file": str(
                destination.relative_to(OUTPUT)
            )
        })


# ======================================================
# METADATA
# ======================================================

metadata_path = OUTPUT / "metadata.csv"

with open(
    metadata_path,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "dataset",
            "origin",
            "file"
        ]
    )

    writer.writeheader()
    writer.writerows(metadata)


# ======================================================
# RESULTADO
# ======================================================

print("\n==============================")
print("CROSS FINALIZADO")
print("==============================")
print(f"Classes: {len(classes)}")
print(f"Vídeos: {len(metadata)}")
print(f"Pasta: {OUTPUT}")
print(f"Metadata: {metadata_path}")