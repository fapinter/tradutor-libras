"""
Separa o MINDS (dataset de treino, 8 intérpretes) em treino e teste por intérprete,
para avaliar o modelo apenas no MINDS com intérpretes nunca vistos no treino.

O split antigo era por vídeo: como cada intérprete grava 5 repetições de cada sinal, o teste tinha
outras repetições dos mesmos intérpretes do treino e as métricas saíam otimistas.

Divisão padrão (INTERPRETES_TESTE_MINDS):
- teste: intérpretes 5 e 8;
- validação (Early Stopping, INTERPRETES_VALIDACAO): intérpretes 1 e 6, que ficam no CSV de treino
  e são separados pelo pipeline.py;
- treino: intérpretes 2, 3, 4 e 7.

Os intérpretes de teste podem ser passados na linha de comando (python split_minds.py 2 3), para
repetir a avaliação com outros pares de teste e reportar a média.
"""
import sys

import pandas as pd

from utils.constants import (
    DATASET_TREINO_CSV,
    DATASET_TREINO_MINDS_CSV,
    DATASET_TESTE_MINDS_CSV,
    INTERPRETES_TESTE_MINDS,
    INTERPRETES_VALIDACAO,
)
from utils.utils import interprete_do_video

interpretes_teste = tuple(int(i) for i in sys.argv[1:]) or INTERPRETES_TESTE_MINDS
if set(interpretes_teste) & set(INTERPRETES_VALIDACAO):
    raise ValueError(f"Os intérpretes de teste {interpretes_teste} não podem estar na validação {INTERPRETES_VALIDACAO}")

df = pd.read_csv(DATASET_TREINO_CSV)
interprete = df["video_id"].map(interprete_do_video)

# divide por intérprete: todos os vídeos de um intérprete ficam no mesmo conjunto
eh_teste = interprete.isin(interpretes_teste)
df_treino = df[~eh_teste].copy()
df_teste = df[eh_teste].copy()

# salva os novos CSVs
df_treino.to_csv(DATASET_TREINO_MINDS_CSV, index=False)
df_teste.to_csv(DATASET_TESTE_MINDS_CSV, index=False)

for nome, df_conjunto, interpretes in (
    ("TREINO", df_treino, interprete[~eh_teste]),
    ("TESTE", df_teste, interprete[eh_teste]),
):
    print(f"\n{nome}")
    print(f"Intérpretes: {sorted(interpretes.unique().tolist())}")
    print(f"Vídeos: {df_conjunto['video_id'].nunique()}")
    print(f"Amostras: {df_conjunto['sample_idx'].nunique()}")
    print(f"Gestos: {df_conjunto['target'].nunique()}")
    print(f"Linhas: {len(df_conjunto)}")
print(f"\nValidação (separada do treino pelo pipeline.py): intérpretes {list(INTERPRETES_VALIDACAO)}")

# garante que nenhum intérprete esteja nos dois conjuntos
overlap = set(interprete[~eh_teste]) & set(interprete[eh_teste])
print(f"\nIntérpretes presentes nos dois conjuntos: {len(overlap)}")
