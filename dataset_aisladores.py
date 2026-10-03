from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class DatasetAisladores(Dataset):
    def __init__(self, ruta_csv, carpeta_imagenes, split):
        tabla = pd.read_csv(ruta_csv)

        self.tabla = tabla[tabla["split"] == split].reset_index(drop=True)
        self.carpeta_imagenes = Path(carpeta_imagenes) / split

    def __len__(self):
        return len(self.tabla)

    def __getitem__(self, indice):
        fila = self.tabla.iloc[indice]
        ruta = self.carpeta_imagenes / fila["filename"]

        with Image.open(ruta) as archivo:
            imagen = archivo.convert("RGB")

        imagen = transformacion(imagen)

        etiquetas = torch.tensor(
            [fila["broken"], fila["pollution-flashover"]],
            dtype=torch.float32,
        )

        return imagen, etiquetas

if __name__ == "__main__":
    train = DatasetAisladores(
        "data/processed/labels.csv",
        "data/processed",
        "train",
    )

    imagen, etiquetas = train[0]

    print("Ejemplos:", len(train))
    print("Forma de la imagen:", imagen.shape)
    print("Etiquetas:", etiquetas)