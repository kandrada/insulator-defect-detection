import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score
from dataset_aisladores import DatasetAisladores


ROOT = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", dispositivo)
    if dispositivo.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train = DatasetAisladores(
        ROOT / "data/processed/labels.csv",
        ROOT / "data/processed",
        "train",
    )
    valid = DatasetAisladores(
        ROOT / "data/processed/labels.csv",
        ROOT / "data/processed",
        "valid",
    )

    cargador_train = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=dispositivo.type == "cuda",
    )
    cargador_valid = DataLoader(
        valid,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=dispositivo.type == "cuda",
    )

    modelo = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    for parametro in modelo.parameters():
        parametro.requires_grad = False

    entradas = modelo.fc.in_features
    modelo.fc = nn.Linear(entradas, 2)

    #Movemos el modelo antes de crear el optimizador
    modelo = modelo.to(dispositivo)

    perdida_fn = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.AdamW(
        modelo.fc.parameters(),
        lr=0.001,
    )

    carpeta = ROOT / "artifacts"
    carpeta.mkdir(exist_ok=True)

    ruta_modelo = carpeta / "resnet18_final.pt"
    ruta_historial = carpeta / "resnet18_historial.csv"

    print("Entradas de la última capa:", entradas)
    print(
        "Parámetros entrenables:",
        sum(p.numel() for p in modelo.parameters() if p.requires_grad),
    )

    historial = []
    mejor_f1 = -1.0
    mejor_epoca = None
    inicio_total = time.perf_counter()

    for epoca in range(1, 6):
        inicio_epoca = time.perf_counter()

        #Conservamos las estadísticas de batchnorm, la ultima capa es la entrenable
        modelo.eval()
        suma_perdidas = 0.0

        for imagenes, etiquetas in cargador_train:
            imagenes = imagenes.to(dispositivo, non_blocking=True)
            etiquetas = etiquetas.to(dispositivo, non_blocking=True)

            optimizador.zero_grad()

            logits = modelo(imagenes)
            perdida = perdida_fn(logits, etiquetas)

            perdida.backward()
            optimizador.step()

            suma_perdidas += perdida.item() * len(imagenes)

        perdida_train = suma_perdidas / len(train)

        verdaderas = []
        predichas = []

        with torch.no_grad():
            for imagenes, etiquetas in cargador_valid:
                imagenes = imagenes.to(
                    dispositivo, non_blocking=True
                )

                logits = modelo(imagenes)
                puntuaciones = torch.sigmoid(logits)
                decisiones = (puntuaciones >= 0.5).int()

                #Las etiquetas ya están en la cpu
                verdaderas.extend(etiquetas.int().tolist())
                predichas.extend(decisiones.cpu().tolist())

        f1_valid = f1_score(
            verdaderas,
            predichas,
            average="macro",
            zero_division=0,
        )

        if f1_valid > mejor_f1:
            mejor_f1 = f1_valid
            mejor_epoca = epoca

            #Guardamos en la cpu para facilitar su carga
            pesos_cpu = {
                nombre: tensor.detach().cpu()
                for nombre, tensor in modelo.state_dict().items()
            }
            torch.save(pesos_cpu, ruta_modelo)

        segundos = time.perf_counter() - inicio_epoca

        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            "macro_f1_valid": f1_valid,
            "segundos": segundos,
        })

        pd.DataFrame(historial).to_csv(ruta_historial, index=False)

        print(
            f"Época {epoca}/5 | "
            f"pérdida train: {perdida_train:.4f} | "
            f"macro-F1 valid: {f1_valid:.4f} | "
            f"tiempo: {segundos:.1f} s",
            flush=True,
        )

    print("\nMejor época:", mejor_epoca)
    print("Mejor macro-F1 validación:", round(mejor_f1, 4))
    print(
        "Tiempo total:",
        round(time.perf_counter() - inicio_total, 1),
        "segundos",
    )
    print("Modelo guardado:", ruta_modelo)
    print(
        f"Tamaño del checkpoint: "
        f"{ruta_modelo.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()