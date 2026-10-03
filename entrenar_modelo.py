import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score
from dataset_aisladores import DatasetAisladores


torch.manual_seed(42)

train = DatasetAisladores(
    "data/processed/labels.csv", "data/processed", "train"
)
valid = DatasetAisladores(
    "data/processed/labels.csv", "data/processed", "valid"
)

cargador_train = DataLoader(
    train, batch_size=32, shuffle=True, num_workers=0
)
cargador_valid = DataLoader(
    valid, batch_size=32, shuffle=False, num_workers=0
)

modelo = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

for parametro in modelo.parameters():
    parametro.requires_grad = False

entradas = modelo.classifier[3].in_features
modelo.classifier[3] = nn.Linear(entradas, 2)

perdida_fn = nn.BCEWithLogitsLoss()
optimizador = torch.optim.AdamW(
    modelo.classifier[3].parameters(),
    lr=0.001,
)

inicio = time.time()
mejor_f1 = -1.0
mejor_epoca = None

for epoca in range(1, 6):
    #las capas preentrenadas permanecen
    modelo.eval()
    suma_perdidas = 0.0

    for imagenes, etiquetas in cargador_train:
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
            logits = modelo(imagenes)
            probabilidades = torch.sigmoid(logits)
            decisiones = (probabilidades >= 0.5).int()

            verdaderas.extend(etiquetas.int().tolist())
            predichas.extend(decisiones.tolist())

    f1 = f1_score(
        verdaderas,
        predichas,
        average="macro",
        zero_division=0,
    )

    print(
        f"Época {epoca}/5 | "
        f"pérdida train: {perdida_train:.4f} | "
        f"macro-F1 valid: {f1:.4f}"
    )

    if f1 > mejor_f1:
        mejor_f1 = f1
        mejor_epoca = epoca

        Path("artifacts").mkdir(exist_ok=True)
        torch.save(
            modelo.state_dict(),
            "artifacts/mobilenet_final.pt",
        )

print("Mejor época:", mejor_epoca)
print("Mejor macro-F1 validación:", round(mejor_f1, 4))
print("Tiempo total:", round(time.time() - inicio, 1), "segundos")