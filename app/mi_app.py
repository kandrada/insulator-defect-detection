import hashlib
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset_aisladores import transformacion


st.set_page_config(
    page_title="Inspección de aisladores",
    page_icon="🌿",
    layout="wide",
)

# Estilo visual: espacios, tarjetas y tipografía.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.035em;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 20px;
    }
    div.stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 600;
    }
    .eyebrow {
        color: #567365;
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .intro {
        color: #617068;
        max-width: 720px;
        font-size: 1.05rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelo():
    modelo = models.mobilenet_v3_small(weights=None)

    entradas = modelo.classifier[3].in_features
    modelo.classifier[3] = nn.Linear(entradas, 2)

    ruta_pesos = ROOT / "artifacts" / "mobilenet_final.pt"
    pesos = torch.load(
        ruta_pesos,
        map_location="cpu",
        weights_only=True,
    )

    modelo.load_state_dict(pesos)
    modelo.eval()
    return modelo


st.markdown(
    '<div class="eyebrow">INSPECCIÓN VISUAL · PROTOTIPO</div>',
    unsafe_allow_html=True,
)
st.title("Una mirada al estado de tus aisladores")
st.markdown(
    '<p class="intro">'
    "Cargá una fotografía, seleccioná un aislador y analizá "
    "posibles señales de rotura o contaminación/flashover."
    "</p>",
    unsafe_allow_html=True,
)

st.caption(
    "Selección manual del aislador · "
    "Dos defectos evaluados de forma independiente"
)

st.divider()

archivo = st.file_uploader(
    "1 · Cargá una fotografía",
    type=["jpg", "jpeg", "png"],
    help="Elegí una foto donde el aislador se vea con claridad.",
)

if archivo is None:
    st.session_state.pop("resultado", None)

    with st.container(border=True):
        st.subheader("Subí una imagen")
        st.write(
            "Subí una fotografía para seleccionar el aislador "
            "que querés revisar."
        )
        st.caption(
            "Incluí el aislador completo y evitá recortes "
            "con demasiado fondo."
        )

else:
    contenido = archivo.getvalue()
    identificador = hashlib.sha256(contenido).hexdigest()

    try:
        with Image.open(BytesIO(contenido)) as original:
            imagen = ImageOps.exif_transpose(original).convert("RGB")
    except (OSError, ValueError):
        st.error("No pudimos abrir la imagen. Probá con otro archivo.")
        st.stop()

    ancho, alto = imagen.size

    seleccion, resultados = st.columns([1.3, 1], gap="large")

    with seleccion:
        with st.container(border=True):
            st.subheader("2 · Seleccioná el aislador")
            st.caption(
                "Mové los dos extremos de cada control "
                "para ajustar el recorte."
            )

            izquierda, derecha = st.slider(
                "Límites horizontales",
                min_value=0,
                max_value=ancho,
                value=(0, ancho),
                key=f"horizontal_{identificador}",
            )

            arriba, abajo = st.slider(
                "Límites verticales",
                min_value=0,
                max_value=alto,
                value=(0, alto),
                key=f"vertical_{identificador}",
            )

            # Dibujamos sobre una copia para no alterar la inferencia.
            vista = imagen.copy()
            dibujo = ImageDraw.Draw(vista)

            if derecha > izquierda and abajo > arriba:
                dibujo.rectangle(
                    (izquierda, arriba, derecha - 1, abajo - 1),
                    outline="#356859",
                    width=max(2, round(max(ancho, alto) / 200)),
                )

            st.image(
                vista,
                caption="El marco verde indica la zona seleccionada",
                use_container_width=True,
            )

    limites = (izquierda, arriba, derecha, abajo)
    firma = (identificador, limites)

    # Una predicción anterior solo sirve para su recorte original.
    resultado_anterior = st.session_state.get("resultado")
    if (
        resultado_anterior is not None
        and resultado_anterior["firma"] != firma
    ):
        st.session_state.pop("resultado", None)

    with resultados:
        with st.container(border=True):
            st.subheader("3 · Analizá el recorte")

            if derecha <= izquierda or abajo <= arriba:
                st.info(
                    "Separá los extremos de los controles "
                    "para seleccionar una zona con ancho y alto."
                )

            else:
                recorte = imagen.crop(limites)

                st.image(
                    recorte,
                    caption="Recorte que recibirá el modelo",
                    width=260,
                )

                analizar = st.button(
                    "Analizar aislador",
                    type="primary",
                    use_container_width=True,
                )

                if analizar:
                    with st.spinner("Analizando el recorte…"):
                        modelo = cargar_modelo()
                        tensor = transformacion(recorte).unsqueeze(0)

                        with torch.no_grad():
                            logits = modelo(tensor)
                            puntuaciones = torch.sigmoid(logits)[0].tolist()

                        st.session_state["resultado"] = {
                            "firma": firma,
                            "puntuaciones": puntuaciones,
                        }

                resultado = st.session_state.get("resultado")

                if resultado is None:
                    st.caption(
                        "Cuando estés conforme con la selección, "
                        "pulsá el botón para ver el resultado."
                    )

                else:
                    st.divider()
                    st.subheader("Resultado")

                    nombres = ["Rotura", "Contaminación/flashover"]
                    puntuaciones = resultado["puntuaciones"]

                    for nombre, puntuacion in zip(
                        nombres, puntuaciones
                    ):
                        st.markdown(
                            f"**{nombre} · {puntuacion:.1%}**"
                        )
                        st.progress(float(puntuacion))
                        st.caption(
                            "Señal marcada"
                            if puntuacion >= 0.5
                            else "Sin señal marcada"
                        )

                    marcados = [
                        nombre.lower()
                        for nombre, puntuacion in zip(
                            nombres, puntuaciones
                        )
                        if puntuacion >= 0.5
                    ]

                    if marcados:
                        st.warning(
                            "Posible defecto: "
                            + " y ".join(marcados)
                            + ". Requiere revisión humana."
                        )
                    else:
                        st.info(
                            "Sin defectos marcados con el umbral actual. "
                            "Esto no garantiza que el aislador esté sano."
                        )


st.divider()

with st.expander("Cómo interpretar este análisis"):
    st.write(
        "El modelo clasifica el recorte seleccionado. "
        "Evalúa rotura y contaminación/flashover de manera "
        "independiente, por lo que puede marcar ambos defectos."
    )
    st.write(
        "Una puntuación igual o superior al 50 % activa la alerta. "
        "El encuadre, la iluminación y la calidad de la imagen "
        "pueden influir en el resultado."
    )


st.caption(
    "Kimberly Andrada · Proyecto final · Diplomado ML/DL · FIUNA 2026"
)