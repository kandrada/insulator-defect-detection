from torch.utils.data import DataLoader
from dataset_aisladores import DatasetAisladores

train = DatasetAisladores(
    "data/processed/labels.csv",
    "data/processed",
    "train",
)

valid = DatasetAisladores(
    "data/processed/labels.csv",
    "data/processed",
    "valid",
)

cargador_train = DataLoader(
    train,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

cargador_valid = DataLoader(
    valid,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

imagenes, etiquetas = next(iter(cargador_train))

print("Lote de imágenes:", imagenes.shape)
print("Lote de etiquetas:", etiquetas.shape)
print("Primeras etiquetas:\n", etiquetas[:5])