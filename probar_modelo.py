import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset_aisladores import DatasetAisladores


train = DatasetAisladores(
    "data/processed/labels.csv",
    "data/processed",
    "train",
)

cargador = DataLoader(
    train,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

modelo = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

# Congelamos las capas preentrenadas.
for parametro in modelo.parameters():
    parametro.requires_grad = False

# Sustituimos la última capa por dos salidas nuevas.
entradas = modelo.classifier[3].in_features
modelo.classifier[3] = nn.Linear(entradas, 2)

imagenes, etiquetas = next(iter(cargador))

modelo.eval()
with torch.no_grad():
    logits = modelo(imagenes)

probabilidades = torch.sigmoid(logits)
perdida = nn.BCEWithLogitsLoss()(logits, etiquetas)

print("Entradas de la última capa:", entradas)
print("Forma de las salidas:", logits.shape)
print("Logits de la primera imagen:", logits[0])
print("Probabilidades de la primera imagen:", probabilidades[0])
print("Etiquetas reales:", etiquetas[0])
print("Pérdida inicial:", perdida.item())