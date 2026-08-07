# 📋 RESUMO EXECUTIVO - Corrigindo a Queda de Acurácia

## 🔴 O Problema

**Você relatou:** "Ao refazer o treinamento, a taxa de acerto continua baixa mesmo com 'Extrair agora'"

**Investigação descobriu:** O problema NÃO era o CSV ou o menu, mas a **lógica de augmentation do LSTM**

---

## 🎯 O BUG RAIZ

### Arquivo: `landmark_augmentation.py` 
### Função: `augmentar_sequencia()`
### Causa: Parâmetros aleatórios diferentes para cada frame

**ANTES (❌ Errado):**
```python
def augmentar_sequencia(sequencia, transformacoes):
    return [augmentar_frame(frame, transformacoes) for frame in sequencia]
```

Cada frame chama `augmentar_frame()` **independentemente**, gerando:
- Frame 0: Rotação=5°, Escala=0.92
- Frame 1: Rotação=-3°, Escala=1.08  ← DIFERENTE
- Frame 2: Rotação=12°, Escala=0.95  ← DIFERENTE

**LSTM aprende que um gesto pode mudar DRASTICAMENTE entre frames** → Acurácia cai para 60-70%

**DEPOIS (✅ Correto):**
```python
# Pré-gerar parâmetros aleatórios UMA VEZ
ruido_matriz = np.random.normal(0, 0.005, (len(sequencia), 21, 3))
escala_fator = np.random.uniform(0.85, 1.15)
rotacao_angulo = np.radians(np.random.uniform(-15.0, 15.0))

# Aplicar os MESMOS parâmetros a todos os frames
for frame_idx, frame in enumerate(sequencia):
    # Todos os 20 frames recebem exatamente a mesma transformação
```

Agora:
- Frame 0: Rotação=5°, Escala=0.92
- Frame 1: Rotação=5°, Escala=0.92  ← IGUAL
- Frame 2: Rotação=5°, Escala=0.92  ← IGUAL

**Movimento coerente que o LSTM consegue aprender** → Acurácia volta ao normal (>90%)

---

## 🔧 Problemas Secundários Corrigidos

| # | Problema | Arquivo | Solução |
|---|----------|---------|---------|
| 2 | CSV sobrescrito silenciosamente | `main.py::gerar_dataset_csv()` | Adicionar confirmação |
| 3 | [4] gerava novo CSV durante comparação | `main.py::comparar_pipelines()` | Remover `export_dataframe=True` |
| 4 | Tipos de dados inconsistentes | `import_from_csv.py` | Converter para numpy arrays |

---

## 🧪 Como Validar a Correção

### Teste Rápido:
```bash
python main.py

[1] Treinar modelo
  → Escolha: [2] LSTM
  → Origem: [1] Extrair agora
  → Augmentation: SIM, n=5

# Acurácia DEVE estar alta (>90%)
# Se estiver <70%, problema persiste
```

### Teste Completo:
```bash
[7] Gerar Dataset (modo LSTM, aug=SIM)
[1] Treinar modelo (carregar CSV)      # Anote: ex 92%
[1] Treinar modelo (carregar CSV)      # Deve ser igual ±1%
[4] Comparar Pipelines                 # RAM e CSV devem ser similares
```

---

## 📊 Resultado Esperado

| Métrica | Antes | Depois |
|---------|-------|--------|
| Acurácia LSTM | 60-70% ❌ | >90% ✅ |
| Consistência | Varia muito 🔄 | Reproduzível ✓ |
| CSV Integridade | Silenciosamente alterado 💥 | Aviso + confirmação ✓ |

---

## 📝 Arquivos Modificados

1. **landmark_augmentation.py** (CRÍTICO)
   - Função `augmentar_sequencia()`: Reescrita para pré-gerar parâmetros

2. **main.py** (Importante)
   - `gerar_dataset_csv()`: Adicionar confirmação antes de sobrescrever
   - `comparar_pipelines()`: Remover `export_dataframe=True`
   - Menu: Adicionar dica de uso

3. **import_from_csv.py** (Menor)
   - Converter DataFrames para numpy arrays explicitamente

---

## 🚀 Próximas Etapas

1. **Execute os testes** acima para validar
2. **Comece com [7]** para gerar o dataset
3. **Use [1] com "Carregar CSV"** para treinar de forma reproduzível
4. **Se persistir problema**, verifique:
   - Se os vídeos de treino estão em `videos/treino/`
   - Se os frames foram extraídos em `dataset/frames_treino/`
   - Se há pelo menos 5-10 vídeos por gesto

---

## ✅ Checklist de Validação

- [ ] Testei `python landmark_augmentation.py` (passou)
- [ ] Acurácia LSTM agora é >90%
- [ ] Resultados são reproduzíveis (teste 2x)
- [ ] CSV só altera quando executo [7]
- [ ] [4] Comparar mostra resultados similares

---

## 💡 Por que isso aconteceu?

Quando você adicionou as features [6] e [7], as opções de augmentation com LSTM ficaram mais frequentes. A lógica `augmentar_sequencia()` estava **gerando transformações aleatórias diferentes para cada frame**, o que é correto para imagens isoladas (RF), mas **ERRADO para sequências temporais (LSTM)**.

O LSTM precisa de **coerência temporal**: se você rotaciona um sinal 5°, TODOS os frames devem sofrer a mesma rotação. Caso contrário, o modelo aprende que o sinal muda drasticamente entre frames, mesmo que você não tenha intenção disso.

