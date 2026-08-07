import sys
import os
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs('logs', exist_ok=True)

from data_preprocessing import extrair_dataset_completo
from model_training import train_random_forest, train_lstm, train_knn
from sign_recognition import recognize_sign
from feature_extraction import extract_features_from_directory
from import_from_csv import import_from_csv


_NOMES_MODELO = {'1': 'Random Forest', '2': 'LSTM', '3': 'KNN'}


def _salvar_log_treino(tipo_modelo, acuracia, n_amostras, augmentar, n_aumentos):
    """Persiste a acurácia de cada treino em logs/resultados_treino.csv."""
    log_path = 'logs/resultados_treino.csv'
    cabecalho = ['data_hora', 'modelo', 'acuracia_%', 'amostras_treino', 'augmentation', 'n_aumentos']
    novo_arquivo = not os.path.exists(log_path)

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(cabecalho)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            _NOMES_MODELO.get(tipo_modelo, tipo_modelo),
            f'{acuracia * 100:.2f}',
            n_amostras,
            'sim' if augmentar else 'nao',
            n_aumentos if augmentar else 0,
        ])
    print(f"[LOG] Resultado salvo em '{log_path}'")


def executar_extracao_de_frames():
    print("\n--- EXTRACAO AUTOMATICA DE VIDEOS ---")
    print("\n[Etapa 1/2] Extraindo pasta de TREINAMENTO...")
    extrair_dataset_completo("videos/treino", "dataset/frames_treino")
    print("\n[Etapa 2/2] Extraindo pasta de TESTE/VALIDACAO...")
    extrair_dataset_completo("videos/teste", "dataset/frames_teste")
    print("\nProcesso de extracao em lote finalizado!")


def _perguntar_augmentation():
    """Pergunta ao usuario se quer augmentation e quantas variacoes."""
    usar = input("\nUsar data augmentation nos landmarks? (s/n) [recomendado com poucos videos]: ").strip().lower()
    if usar == 's':
        try:
            n = int(input("Quantas variacoes por amostra? [padrao: 5, recomendado: 5-10]: ").strip() or "5")
        except ValueError:
            n = 5
        print(f"Augmentation ativada: {n} variacoes por amostra (~{n + 1}x mais dados)")
        print("IMPORTANTE: a augmentation sera aplicada APENAS nos dados de treino (apos o split).")
        return True, n
    return False, 0


def gerar_dataset_csv():
    """Gera e salva o dataset SEM augmentation (dados originais puros)."""
    dataset_root = "dataset/frames_treino"
    print("\n--- GERAR DATASET (FEATURES ORIGINAIS) ---")
    print("Para quais modelos você deseja gerar os dados?")
    print("[1] Modelos Estáticos (Random Forest e KNN)")
    print("[2] Modelo Contínuo (LSTM)")
    modo = input("Escolha (1 ou 2): ").strip()

    if modo not in ['1', '2']:
        print("Opção inválida.")
        return

    modo_extracao = "rf" if modo == '1' else "lstm"
    csv_path = f"dataset/dataset_completo_{modo_extracao}.csv"

    if os.path.exists(csv_path):
        print(f"\n⚠️  AVISO: O arquivo {csv_path} já existe!")
        confirmacao = input("Deseja SOBRESCREVER com novos dados? (s/n): ").strip().lower()
        if confirmacao != 's':
            print("Operação cancelada.")
            return

    print(f"\nExtração no modo {modo_extracao.upper()} (sem augmentation — ocorre no treino)...")
    extract_features_from_directory(
        dataset_root,
        mode=modo_extracao,
        export_dataframe=True,
    )
    print(f"\n[OK] Dataset salvo em '{csv_path}'!")
    print("DICA: ao treinar com este CSV, escolha 'usar augmentation' para dados mais robustos.")


def executar_treinamento():
    """Lógica de escolha da fonte de dados e modelo de treinamento"""
    dataset_root = "dataset/frames_treino"

    print("\n--- INICIANDO PROCESSO DE TREINAMENTO ---")
    print("1. Escolha a arquitetura do modelo primeiro:")
    print("[1] Random Forest (Reconhecimento estático frame-a-frame)")
    print("[2] LSTM (Reconhecimento contínuo de movimento)")
    print("[3] KNN (Reconhecimento estático frame-a-frame)")

    tipo_modelo = input("\nEscolha (1, 2 ou 3): ").strip()

    if tipo_modelo not in ['1', '2', '3']:
        print("Opção inválida. Cancelando treinamento.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"

    print("\n2. Origem dos dados de treinamento:")
    print("[1] Extrair agora (Processar as imagens na hora, sem salvar)")
    print("[2] Carregar Dataset pré-gerado (Ler o arquivo CSV salvo)")
    origem = input("Escolha (1 ou 2): ").strip()

    if origem == '1':
        print(f"\nExtraindo features no modo {modo_extracao.upper()}...")
        features, labels = extract_features_from_directory(
            dataset_root,
            mode=modo_extracao,
        )
    elif origem == '2':
        csv_path = f"dataset/dataset_completo_{modo_extracao}.csv"
        if not os.path.exists(csv_path):
            print(f"\nERRO: O arquivo {csv_path} não foi encontrado.")
            print("Você precisa rodar a opção 'Gerar Dataset' primeiro!")
            return
        print(f"\nCarregando dados do arquivo {csv_path}...")
        features, labels = import_from_csv(csv_path, mode=modo_extracao)
    else:
        print("Opção inválida.")
        return

    # Augmentation perguntada AQUI — será aplicada dentro do treino, após o split
    augmentar, n_aumentos = _perguntar_augmentation()

    print("\n3. Iniciando Treinamento do Modelo...")
    acc = None
    if tipo_modelo == '1':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do Random Forest.")
        else:
            acc = train_random_forest(features, labels, augmentar=augmentar, n_aumentos=n_aumentos, return_accuracy=True)

    elif tipo_modelo == '2':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do LSTM.")
        else:
            acc = train_lstm(features, labels, augmentar=augmentar, n_aumentos=n_aumentos, return_accuracy=True)

    elif tipo_modelo == '3':
        if len(features) < 2:
            print("Erro: Dados insuficientes para treino do KNN.")
        else:
            acc = train_knn(features, labels, augmentar=augmentar, n_aumentos=n_aumentos, return_accuracy=True)

    if acc is not None:
        _salvar_log_treino(tipo_modelo, acc, len(features), augmentar, n_aumentos)


def comparar_pipelines():
    dataset_root = "dataset/frames_treino"

    print("\n--- COMPARACAO DE PIPELINES ---")
    print("[1] Random Forest  [2] LSTM  [3] KNN")
    tipo_modelo = input("\nEscolha o modelo (1, 2 ou 3): ").strip()

    if tipo_modelo not in ['1', '2', '3']:
        print("Opcao invalida.")
        return

    modo_extracao = "rf" if tipo_modelo in ['1', '3'] else "lstm"
    augmentar, n_aumentos = _perguntar_augmentation()

    print(f"\n1. Extraindo features em memória no modo {modo_extracao.upper()}...")
    features_direto, labels_direto = extract_features_from_directory(
        dataset_root,
        mode=modo_extracao,
        export_dataframe=False,
    )

    if len(features_direto) < 2:
        print("Erro: dados insuficientes.")
        return

    nome_padrao = f"dataset/dataset_completo_{modo_extracao}.csv"
    csv_path = input(f"\nCaminho do CSV para comparação [{nome_padrao}]: ").strip() or nome_padrao

    if not os.path.exists(csv_path):
        print(f"\n❌ ERRO: {csv_path} não encontrado!")
        print(f"Você precisa gerar o CSV primeiro usando a opção [7]!")
        return

    print(f"\n2. Carregando dados do CSV...")
    X_csv, y_csv = import_from_csv(csv_path, mode=modo_extracao)

    print("\n3. Treinando pipeline direto (dados em RAM)...")
    if tipo_modelo == '1':
        acc_direto = train_random_forest(
            features_direto, labels_direto,
            model_path="models/model_direto.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )
    elif tipo_modelo == '2':
        acc_direto = train_lstm(
            features_direto, labels_direto,
            model_path="models/lstm_direto.h5",
            encoder_path="models/encoder_direto.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )
    elif tipo_modelo == '3':
        acc_direto = train_knn(
            features_direto, labels_direto,
            model_path="models/knn_direto.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )

    print("\n4. Treinando pipeline via CSV...")
    if tipo_modelo == '1':
        acc_csv = train_random_forest(
            X_csv, y_csv,
            model_path="models/model_csv.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )
    elif tipo_modelo == '2':
        acc_csv = train_lstm(
            X_csv, y_csv,
            model_path="models/lstm_csv.h5",
            encoder_path="models/encoder_csv.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )
    elif tipo_modelo == '3':
        acc_csv = train_knn(
            X_csv, y_csv,
            model_path="models/knn_csv.pkl",
            return_accuracy=True,
            augmentar=augmentar, n_aumentos=n_aumentos,
        )

    print("\n" + "=" * 30)
    print("--- RESULTADO ---")
    print(f"Acuracia Direto:   {acc_direto * 100:.2f}%")
    print(f"Acuracia via CSV:  {acc_csv * 100:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("   SISTEMA DE RECONHECIMENTO DE SINAIS (LIBRAS)")
        print("=" * 50)
        print("[1] Treinar modelo")
        print("[2] Testar via Webcam")
        print("[3] Testar via Video (pasta de teste)")
        print("[4] Comparar pipeline direto vs CSV")
        print("[5] Extrair frames de videos em lote")
        print("[6] Gerar Matriz de Confusão")
        print("[7] Gerar Dataset (Salvar CSV)")
        print("[0] Sair do programa")
        print("=" * 50)
        print("\n DICA: Opção [7] gera o CSV com dados originais.")
        print("       A augmentation ocorre automaticamente no treino [1].")
        print("=" * 50)

        escolha = input("\nEscolha uma opção (0 a 7): ").strip()

        if escolha == '1':
            executar_treinamento()

        elif escolha == '2':
            print("\n[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Modelo (1, 2 ou 3): ").strip()
            recognize_sign(0, tipo_modelo=modelo_teste)

        elif escolha == '3':
            print("\n[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_teste = input("Modelo (1, 2 ou 3): ").strip()
            pasta_teste = input("Pasta dos videos de teste [videos/teste]: ").strip() or "videos/teste"

            if os.path.exists(pasta_teste) and os.path.isdir(pasta_teste):
                videos_encontrados = []
                for root, dirs, files in os.walk(pasta_teste):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov')):
                            videos_encontrados.append(os.path.join(root, file))

                if not videos_encontrados:
                    print(f"Nenhum video encontrado em '{pasta_teste}'.")
                else:
                    print(f"{len(videos_encontrados)} video(s) encontrado(s).")

                    resultados_salvos = []

                    for caminho_video in videos_encontrados:
                        print(f"\n-> Analisando: {caminho_video}")
                        predicao_final = recognize_sign(caminho_video, tipo_modelo=modelo_teste)

                        if predicao_final:
                            resultados_salvos.append(predicao_final)

                    os.makedirs("outputs", exist_ok=True)
                    arquivo_txt = os.path.join("outputs", f"predicoes_modelo_{modelo_teste}.txt")
                    with open(arquivo_txt, "w", encoding="utf-8") as f:
                        for resultado in resultados_salvos:
                            f.write(resultado + "\n")

                    print(f"\n[OK] Bateria de testes concluída!")
                    print(f"As predições foram salvas no arquivo '{arquivo_txt}'.")
            else:
                print(f"Diretorio '{pasta_teste}' nao encontrado.")

        elif escolha == '4':
            comparar_pipelines()

        elif escolha == '5':
            executar_extracao_de_frames()

        elif escolha == '6':
            print("\n--- GERAR MATRIZ DE CONFUSÃO ---")
            print("[1] Random Forest  [2] LSTM  [3] KNN")
            modelo_matriz = input("De qual algoritmo você deseja ler os resultados? (1, 2 ou 3): ").strip()

            arquivo_txt = os.path.join("outputs", f"predicoes_modelo_{modelo_matriz}.txt")

            if not os.path.exists(arquivo_txt):
                print(f"\nErro: Arquivo '{arquivo_txt}' não encontrado.")
                print("Você precisa rodar a bateria de testes (Opção 3) para este modelo primeiro!")
                continue

            with open(arquivo_txt, "r", encoding="utf-8") as f:
                y_pred = [linha.strip() for linha in f.readlines() if linha.strip()]

            print(f"\nForam encontradas {len(y_pred)} predições salvas deste modelo.")

            gabarito_file = os.path.join("outputs", "gabarito.txt")
            y_true_gabarito = []
            if os.path.exists(gabarito_file):
                with open(gabarito_file, "r", encoding="utf-8") as f:
                    conteudo = f.read().strip()
                    if "," in conteudo:
                        y_true_gabarito = [item.strip() for item in conteudo.split(",") if item.strip()]
                    else:
                        y_true_gabarito = [linha.strip() for linha in conteudo.splitlines() if linha.strip()]

            if y_true_gabarito:
                print(f"\n[INFO] Gabarito encontrado em '{gabarito_file}' ({len(y_true_gabarito)} rótulos).")
                lista_usuario = input(f"Pressione ENTER para carregar de '{gabarito_file}' ou digite a lista manualmente: ").strip()
                if not lista_usuario:
                    y_true = y_true_gabarito
                else:
                    y_true = [item.strip() for item in lista_usuario.split(",") if item.strip()]
            else:
                print("\nDigite a lista com as classes REAIS na ordem em que os vídeos foram analisados.")
                print("Formato esperado: separado por vírgulas (ex: ola, ola, obrigado, desculpa)")
                lista_usuario = input("Sua lista de gabarito: ").strip()
                if not lista_usuario:
                    print("Operação cancelada.")
                    continue
                y_true = [item.strip() for item in lista_usuario.split(",") if item.strip()]

            if len(y_true) != len(y_pred):
                print(f"\nERRO DE TAMANHO!")
                print(f"Sua lista tem {len(y_true)} rótulos, mas o arquivo tem {len(y_pred)} vídeos processados.")
                continue

            classes_unicas = sorted(list(set(y_true + y_pred)))

            print("\nGerando gráfico...")
            cm = confusion_matrix(y_true, y_pred, labels=classes_unicas)

            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_unicas)
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format='d')
            plt.xticks(ha='right')
            ax.tick_params(axis='both', which='major', labelsize=10)
            nome_modelo = _NOMES_MODELO.get(modelo_matriz, modelo_matriz)
            plt.title(f"Matriz de Confusão - {nome_modelo}", fontsize=14, pad=20)
            plt.xlabel('Sinal Traduzido (Predição)', fontsize=12, labelpad=10)
            plt.ylabel('Sinal Real (Gabarito)', fontsize=12, labelpad=10)
            plt.tight_layout()

            figura_path = f"logs/matriz_confusao_modelo_{modelo_matriz}.png"
            plt.savefig(figura_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Figura salva em '{figura_path}'")
            plt.show()

        elif escolha == '7':
            gerar_dataset_csv()

        elif escolha == '0':
            print("\nEncerrando. Ate logo!")
            break

        else:
            print("\nOpcao invalida.")
