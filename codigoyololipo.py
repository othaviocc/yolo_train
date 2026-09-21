from pathlib import Path
import yaml
from ultralytics import YOLO
import os

# Como os diretórios train/ e valid/ estão na mesma pasta do script, a raiz é o diretório atual (".")
DATASET_ROOT = os.path.abspath(".")  
CLASS_NAMES = {
    0: "lipo",  # Atualizado para bater com o label exportado do Roboflow
}

DATA_YAML_PATH = "data.yaml"

def gerar_data_yaml():
    """Cria (ou atualiza) o data.yaml para bater com a estrutura do Roboflow."""
    config = {
        "path": DATASET_ROOT,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images", # Adicionado pois você tem uma pasta test
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] {DATA_YAML_PATH} gerado/sobrescrito com sucesso.")

def treinar():
    # Carrega o modelo pre-treinado YOLOv11 nano (baixa sozinho na 1a vez)
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML_PATH,
        epochs=150,
        imgsz=960,               # resolucao maior ajuda se as caixinhas forem
                                  # pequenas dentro da imagem. Se o treino
                                  # ficar muito lento/pesado, reduza para 640.
        batch=16,                 # reduza se faltar memoria de GPU (ex: 8)
        patience=30,              # para o treino cedo se nao melhorar por
                                  # 30 epocas seguidas (evita overfitting)
        device="cpu",                 # 0 = primeira GPU. Use "cpu" se nao tiver GPU.

        # --- Cor: reduzido, pois o objeto e cinza (pouca info de cor) ---
        hsv_h=0.010,
        hsv_s=0.30,
        hsv_v=0.40,

        # --- Geometria: mantido forte, e o que mais ajuda aqui ---
        degrees=10.0,
        translate=0.10,
        scale=0.50,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,               # 0 = nao inverte de cabeca para baixo
        fliplr=0.5,               # 0.5 = espelha horizontalmente metade das vezes

        mosaic=1.0,               # combina 4 imagens em 1 (bom p/ objetos pequenos)
        mixup=0.10,
        close_mosaic=10,          # desliga o mosaic nas ultimas 10 epocas

        project="runs_caixinhas",
        name="yolo11n_caixinhas",
        exist_ok=True,            # permite rodar de novo sem dar erro de pasta existente
    )

    print("\n[OK] Treino concluido. Validando modelo final...")
    metrics = model.val()
    print(metrics)

    print(
        "\nModelo final salvo em: "
        "runs_caixinhas/yolo11n_caixinhas/weights/best.pt\n"
        "Use esse caminho no predict.py para testar em novas imagens."
    )


if __name__ == "__main__":
    gerar_data_yaml()
    treinar()