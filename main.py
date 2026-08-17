import sys
import os
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.makedirs('logs', exist_ok=True)

from data_preprocessing import extrair_dataset_completo
from model_training import train_lstm
from feature_extraction import extract_features_from_directory, import_from_csv


def _salvar_log_treino(acuracia, n_amostras, augmentar, n_aumentos):
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
            'LSTM',
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
    """Gera e salva o dataset SEM augmentation (dados originais puros) para o LSTM."""
    print("\n--- GERAR DATASET LSTM (FEATURES ORIGINAIS) ---")
    print("[1] Treino (dataset/frames_treino -> dataset/dataset_completo_lstm.csv)")
    print("[2] Teste (dataset/frames_teste -> dataset/dataset_teste_lstm.csv)")
    print("[3] Ambos")
    opcao = input("Escolha uma opção (1, 2 ou 3) [padrão: 1]: ").strip() or "1"

    alvos = []
    if opcao == '1':
        alvos.append(("dataset/frames_treino", "dataset/dataset_completo_lstm.csv", "Treino"))
    elif opcao == '2':
        alvos.append(("dataset/frames_teste", "dataset/dataset_teste_lstm.csv", "Teste"))
    elif opcao == '3':
        alvos.append(("dataset/frames_treino", "dataset/dataset_completo_lstm.csv", "Treino"))
        alvos.append(("dataset/frames_teste", "dataset/dataset_teste_lstm.csv", "Teste"))
    else:
        print("Opção inválida.")
        return

    for dataset_root, csv_path, rotulo in alvos:
        print(f"\n--- Processando conjunto de {rotulo.upper()} ---")
        if not os.path.exists(dataset_root):
            print(f"⚠️  Diretório '{dataset_root}' não encontrado.")
            print(f"Dica: Execute a opção [2] do menu principal para extrair os frames primeiro.")
            continue

        if os.path.exists(csv_path):
            print(f"\n⚠️  AVISO: O arquivo {csv_path} já existe!")
            confirmacao = input(f"Deseja SOBRESCREVER '{csv_path}'? (s/n): ").strip().lower()
            if confirmacao != 's':
                print(f"Extração de {rotulo} cancelada.")
                continue

        print(f"\nExtraindo features de '{dataset_root}'...")
        extract_features_from_directory(
            dataset_root,
            mode="lstm",
            export_dataframe=True,
            output_csv_path=csv_path,
        )
        print(f"[OK] Dataset de {rotulo} salvo em '{csv_path}'!")


def executar_treinamento():
    """Treinamento exclusivo do modelo LSTM."""
    dataset_root = "dataset/frames_treino"

    print("\n--- INICIANDO TREINAMENTO DO MODELO LSTM ---")
    print("Origem dos dados de treinamento:")
    print("[1] Extrair agora (Processar as imagens na hora, sem salvar)")
    print("[2] Carregar Dataset pré-gerado (Ler o arquivo CSV salvo)")
    origem = input("Escolha (1 ou 2): ").strip()

    if origem == '1':
        print("\nExtraindo features para o modelo LSTM...")
        features, labels = extract_features_from_directory(
            dataset_root,
            mode="lstm",
        )
    elif origem == '2':
        csv_path = "dataset/dataset_completo_lstm.csv"
        if not os.path.exists(csv_path):
            print(f"\nERRO: O arquivo {csv_path} não foi encontrado.")
            print("Você precisa rodar a opção 'Gerar Dataset' primeiro!")
            return
        print(f"\nCarregando dados do arquivo {csv_path}...")
        features, labels = import_from_csv(csv_path, mode="lstm")
    else:
        print("Opção inválida.")
        return

    augmentar, n_aumentos = _perguntar_augmentation()

    print("\nIniciando Treinamento do Modelo LSTM...")
    if len(features) < 2:
        print("Erro: Dados insuficientes para treino do LSTM.")
        return

    acc = train_lstm(features, labels, augmentar=augmentar, n_aumentos=n_aumentos, return_accuracy=True)

    if acc is not None:
        _salvar_log_treino(acc, len(features), augmentar, n_aumentos)


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("   SISTEMA DE RECONHECIMENTO DE LIBRAS (LSTM)")
        print("=" * 50)
        print("[1] Treinar modelo LSTM")
        print("[2] Extrair frames de vídeos em lote")
        print("[3] Gerar Dataset (Salvar CSV)")
        print("[0] Sair do programa")
        print("=" * 50)
        print("\n DICA: Para testes ao vivo em tempo real com a câmera,")
        print("       execute diretamente o script 'live_testing.py'.")
        print("=" * 50)

        escolha = input("\nEscolha uma opção (0 a 3): ").strip()

        if escolha == '1':
            executar_treinamento()

        elif escolha == '2':
            executar_extracao_de_frames()

        elif escolha == '3':
            gerar_dataset_csv()

        elif escolha == '0':
            print("\nEncerrando. Até logo!")
            break

        else:
            print("\nOpção inválida.")
