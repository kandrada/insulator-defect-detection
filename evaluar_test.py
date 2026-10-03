import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, f1_score
from dataset_aisladores import DatasetAisladores


test = DatasetAisladores(
    "data/processed/labels.csv",
    "data/processed",
    "test",
)

cargador_test = DataLoader(
    test,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

modelo = models.mobilenet_v3_small(weights=None)
entradas = modelo.classifier[3].in_features
modelo.classifier[3] = nn.Linear(entradas, 2)

pesos = torch.load(
    "artifacts/mobilenet_final.pt",
    map_location="cpu",
    weights_only=True,
)
modelo.load_state_dict(pesos)
modelo.eval()

verdaderas = []
predichas = []

with torch.no_grad():
    for imagenes, etiquetas in cargador_test:
        logits = modelo(imagenes)
        probabilidades = torch.sigmoid(logits)
        decisiones = (probabilidades >= 0.5).int()

        verdaderas.extend(etiquetas.int().tolist())
        predichas.extend(decisiones.tolist())

print("Recortes test:", len(test))
print(
    classification_report(
        verdaderas,
        predichas,
        target_names=["broken", "pollution-flashover"],
        digits=3,
        zero_division=0,
    )
)
print(
    "Macro-F1 test:",
    round(f1_score(verdaderas, predichas, average="macro", zero_division=0), 4),
)