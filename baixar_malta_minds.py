from pathlib import Path
from collections import defaultdict
import csv
import re
import unicodedata
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = Path(__file__).resolve().parent
MALTA = BASE / "MALTA"

MATCHED_CSV = MALTA / "dataset_intersections" / "MINDS_matched_labels.csv"
LINKS_DIR = MALTA / "video_downloads"

OUTPUT = BASE / "MALTA_MINDS"
OUTPUT.mkdir(exist_ok=True)

SOURCES = {
    "Acessibilidades3": {
        "csv": "links_videos_acessibilidade_brasil.csv",
        "label_col": "Palavra",
        "url_col": "Link",
        "allow_number_suffix": True,
    },
    "SpreadTheSign": {
        "csv": "links_videos_spreadthesign.csv",
        "label_col": "Palavra",
        "url_col": "Link",
        "allow_number_suffix": False,
    },
    "UFPE": {
        "csv": "links_videos_vlibrasil.csv",
        "label_col": "sign",
        "url_col": "link",
        "allow_number_suffix": False,
    },
    "UFSC": {
        "csv": "links_videos_ufsc_signbank.csv",
        "label_col": "Palavra",
        "url_col": "Link",
        "allow_number_suffix": True,
    },
    "UFV": {
        "csv": "links_videos_ufv.csv",
        "label_col": "Palavra",
        "url_col": "Link",
        "allow_number_suffix": False,
    },
    "USP": {
        "csv": "links_videos_usp.csv",
        "label_col": "Palavra",
        "url_col": "Link",
        "allow_number_suffix": False,
    },
}


def normalize(text):
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def label_matches(source_label, target_label, allow_number_suffix=False):
    source = normalize(source_label)
    target = normalize(target_label)

    # Ex.: Maçã -> maca
    if source == target:
        return True

    # Exceção usada no MALTA:
    # vacina também aparece como vacinação
    aliases = {
        "vacina": {"vacinacao", "vacinar"},
    }

    if source in aliases.get(target, set()):
        return True

    # Acessibilidade / UFSC:
    # acontecer, acontecer2 etc.
    if allow_number_suffix:
        if re.fullmatch(re.escape(target) + r"\d+", source):
            return True

    return False


# ------------------------------------------------------
# 1. Descobrir quantos vídeos queremos de cada fonte/classe
# ------------------------------------------------------

wanted = defaultdict(lambda: defaultdict(int))

with open(MATCHED_CSV, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        label = normalize(row["name"])
        source = row["dictionary"].strip()

        # Não vamos baixar:
        # Acessibilidades2 -> links antigos .wmv não disponíveis no CSV atual
        # UFSC_V2 -> duplicaria os mesmos vídeos brutos do UFSC
        if source in ("Acessibilidades2", "UFSC_V2"):
            continue

        if source in SOURCES:
            wanted[source][label] += 1


print("\n=== Vídeos planejados ===")

total_wanted = 0

for source, labels in wanted.items():
    subtotal = sum(labels.values())
    total_wanted += subtotal
    print(f"{source}: {subtotal}")

print(f"\nTotal esperado: {total_wanted}")


# ------------------------------------------------------
# 2. Ler os CSVs públicos e localizar URLs
# ------------------------------------------------------

selected = []

for source, labels in wanted.items():

    config = SOURCES[source]
    csv_path = LINKS_DIR / config["csv"]

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for target_label, quantidade in labels.items():

        candidates = []

        for row in rows:
            source_label = row[config["label_col"]]

            if label_matches(
                source_label,
                target_label,
                config["allow_number_suffix"],
            ):
                url = row[config["url_col"]].strip()

                if url and url not in [x["url"] for x in candidates]:
                    candidates.append({
                        "url": url,
                        "original_label": source_label,
                    })

        if len(candidates) < quantidade:
            print(
                f"ATENÇÃO: {source}/{target_label}: "
                f"esperados {quantidade}, encontrados {len(candidates)}"
            )

        for candidate in candidates[:quantidade]:

            selected.append({
                "label": target_label,
                "source": source,
                "url": candidate["url"],
                "original_label": candidate["original_label"],
            })


print(f"\nURLs encontradas: {len(selected)}")


# ------------------------------------------------------
# 3. Remover URLs duplicadas
# ------------------------------------------------------

unique = []
seen_urls = set()

for item in selected:

    if item["url"] in seen_urls:
        continue

    seen_urls.add(item["url"])
    unique.append(item)


print(f"URLs únicas: {len(unique)}")


# ------------------------------------------------------
# 4. Download
# ------------------------------------------------------

manifest = []

for number, item in enumerate(unique, start=1):

    label = item["label"]
    source = item["source"]
    url = item["url"]

    class_dir = OUTPUT / label
    class_dir.mkdir(parents=True, exist_ok=True)

    extension = ".mp4"

    filename = (
        f"malta_{source.lower()}_"
        f"{number:03d}{extension}"
    )

    destination = class_dir / filename

    if destination.exists() and destination.stat().st_size > 0:

        print(f"[OK] Já existe: {destination}")

        manifest.append({
            **item,
            "file": str(destination.relative_to(BASE)),
            "status": "existing",
        })

        continue

    print(
        f"[{number}/{len(unique)}] "
        f"{label} - {source}"
    )

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=60,
            verify=False,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        with open(destination, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    f.write(chunk)

        manifest.append({
            **item,
            "file": str(destination.relative_to(BASE)),
            "status": "downloaded",
        })

        print(
            f"     OK - "
            f"{destination.stat().st_size / 1024 / 1024:.1f} MB"
        )

    except Exception as error:

        print(f"     ERRO: {error}")

        if destination.exists():
            destination.unlink()

        manifest.append({
            **item,
            "file": "",
            "status": f"error: {error}",
        })


# ------------------------------------------------------
# 5. Manifest
# ------------------------------------------------------

manifest_path = OUTPUT / "manifest_malta.csv"

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "source",
            "original_label",
            "url",
            "file",
            "status",
        ]
    )

    writer.writeheader()
    writer.writerows(manifest)


success = sum(
    1
    for item in manifest
    if item["status"] in ("downloaded", "existing")
)

errors = len(manifest) - success

print("\n==========================")
print("DOWNLOAD FINALIZADO")
print("==========================")
print(f"Vídeos OK: {success}")
print(f"Erros: {errors}")
print(f"Pasta: {OUTPUT}")
print(f"Manifest: {manifest_path}")