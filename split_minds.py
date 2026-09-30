import pandas as pd
from sklearn.model_selection import train_test_split

CSV_MINDS = "dataset/treino.csv"

df = pd.read_csv(CSV_MINDS)

videos = df[["video_id", "target"]].drop_duplicates()

# divide por vídeo, mantendo a proporção das classes
videos_treino, videos_teste = train_test_split(
    videos,
    test_size=0.30,
    random_state=42,
    stratify=videos["target"]
)

# recupera todas as linhas/sequências dos vídeos selecionados
df_treino = df[
    df["video_id"].isin(videos_treino["video_id"])
].copy()

df_teste = df[
    df["video_id"].isin(videos_teste["video_id"])
].copy()

# salva os novos CSVs
df_treino.to_csv(
    "dataset/treino_minds_70.csv",
    index=False
)

df_teste.to_csv(
    "dataset/teste_minds_30.csv",
    index=False
)

print("TREINO")
print(f"Vídeos: {df_treino['video_id'].nunique()}")
print(f"Amostras: {df_treino['sample_idx'].nunique()}")
print(f"Linhas: {len(df_treino)}")

print("\nTESTE")
print(f"Vídeos: {df_teste['video_id'].nunique()}")
print(f"Amostras: {df_teste['sample_idx'].nunique()}")
print(f"Linhas: {len(df_teste)}")

# garante que nenhum vídeo esteja nos dois conjuntos
overlap = set(df_treino["video_id"]) & set(df_teste["video_id"])

print(f"\nVídeos presentes nos dois conjuntos: {len(overlap)}")