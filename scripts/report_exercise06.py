"""Informe y documentación del Ejercicio 06 desde los resultados ejecutados."""

from pathlib import Path
import json, shutil
import pandas as pd
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from matplotlib.font_manager import findfont, FontProperties
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/delaxis1/INF8239_U03_Generativos"
NAMES = {"ae": "AE · latente 16", "vae_b1": "VAE · β=1", "vae_b01": "VAE · β=0.1"}
for name, fn, weight, style in [
    ("Body", "calibri.ttf", "normal", "normal"),
    ("Body-Bold", "calibrib.ttf", "bold", "normal"),
    ("Body-Italic", "calibrii.ttf", "normal", "oblique"),
]:
    f = Path("C:/Windows/Fonts") / fn
    pdfmetrics.registerFont(
        TTFont(
            name,
            str(f)
            if f.exists()
            else findfont(
                FontProperties(family="DejaVu Sans", weight=weight, style=style)
            ),
        )
    )
pdfmetrics.registerFontFamily(
    "Body",
    normal="Body",
    bold="Body-Bold",
    italic="Body-Italic",
    boldItalic="Body-Bold",
)


def main():
    out = ROOT / "reports"
    meta = json.loads((out / "metrics.json").read_text())
    t = pd.read_csv(out / "main_results.csv").set_index("model")
    choice = meta["chosen"]
    selected = t.loc[choice]
    ae = t.loc["ae"]
    div = meta["diversity"]
    better = "menor" if ae.test_mse < selected.test_mse else "mayor"
    conclusion = f"""# Conclusión personal — borrador para revisión

Esta práctica me ayudó a separar dos resultados que al principio podían parecer lo mismo: reconstruir una prenda y generar una imagen nueva. El autoencoder recibe una imagen concreta e intenta conservarla después de comprimirla. El VAE añade una distribución latente y una penalización KL para poder explorar ese espacio mediante muestreo. Por eso decidí mirar las reconstrucciones y las muestras del prior por separado, sin presentar una muestra nueva como si correspondiera a la imagen de prueba que aparece encima.

El autoencoder obtuvo MSE {ae.test_mse:.5f}, {better} que el {selected.test_mse:.5f} del VAE seleccionado. Sus SSIM fueron {ae.test_ssim:.4f} y {selected.test_ssim:.4f}, respectivamente. Estas cifras describen reconstrucción de las mismas diez mil imágenes de prueba. No permiten atribuir toda la diferencia a KL: la base también usa arquitecturas y tamaños latentes distintos, dieciséis dimensiones en AE y dos en VAE. Mantener esa distinción me parece necesario para no sacar una conclusión más amplia que el experimento.

Comparé dos pesos de regularización con la misma semilla, partición y presupuesto. La validación seleccionó {NAMES[choice]}, utilizando reconstrucción sumada más KL con peso común uno. Registré esa decisión antes de evaluar prueba. Elegir por las pérdidas totales originales habría sido confuso, porque cada beta cambia la escala del término KL. Las curvas separadas permiten ver qué parte del objetivo cambia durante el entrenamiento y evitan confundir una pérdida baja con una buena generación.

Para la parte generativa utilicé doscientas cincuenta y seis muestras con semilla fija. Su diversidad media en píxeles fue {div["generated_pairwise_mse"]:.5f}, frente a {div["real_validation_pairwise_mse"]:.5f} en la referencia real. Esa comparación muestra variación, pero no mide cobertura de clases ni calidad humana. También busqué vecinos entre las cincuenta y cuatro mil imágenes de ajuste. La mediana de distancia fue {div["generated_nn_mse_median"]:.5f}. Encontrar parecido no demuestra copia; encontrar diferencias tampoco garantiza privacidad. Los paneles y la interpolación sirven para examinar esos límites de forma visible.

Mi decisión es conservar el AE como referencia de reconstrucción y usar el VAE elegido en una demostración local del espacio latente. El experimento completo tomó {meta["total_seconds"] / 60:.1f} minutos y se puede repetir con los archivos, semillas y dependencias registrados. Antes de darle un uso más exigente, repetiría el ajuste con otras semillas, ampliaría la evaluación perceptual y revisaría los ejemplos con personas. Lo entregado es una comparación educativa comprobable, con una sola semilla de entrenamiento y sin garantías de originalidad, equidad o privacidad.
"""
    (out / "conclusion.md").write_text(conclusion, encoding="utf-8")
    table = (
        t[
            [
                "test_mse",
                "test_ssim",
                "test_kl",
                "parameters",
                "train_s",
                "latency_ms",
                "size_kib",
            ]
        ]
        .rename(index=NAMES)
        .to_markdown(floatfmt=".5f")
    )
    (ROOT / "README.md").write_text(
        f"""# INF8239_U03_Generativos

Alexis De La Cruz · 100805370 · INF-8239 · Ejercicio 06.

Base oficial LAB10 de Edwin Ramón José Nolasco (commit acfd6b9). Autoencoder y VAE con Fashion-MNIST completo. [Notebook ejecutado](notebooks/06_ae_vae.ipynb), [informe PDF](reports/informe_ejercicio06.pdf), [Model Card](MODEL_CARD.md), [declaración IA](AI_USE_DECLARATION.md).

## Resultados
Selección de generación en validación: **{NAMES[choice]}**. Prueba: 10,000 imágenes; reconstrucción VAE desde la media latente.

{table}

BCE del AE es media por píxel; BCE VAE se suma en 784 píxeles y KL en 2 dimensiones. Ambos términos se promedian por imagen. Los checkpoints VAE se comparan con un criterio común R+KL (peso1), aunque se entrenan con beta1/0.1. Prueba se consulta después de registrar selección. No se interpreta MSE de reconstrucción como calidad de generación.

## Reproducir (PowerShell)
Desde esta carpeta:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE/.venvs/u03-gen/.venv"
uv sync --frozen --extra cpu --python 3.12.12 --link-mode copy
uv run --extra cpu python -m pytest -q
uv run --extra cpu python scripts/execute_notebook.py
uv run --extra cpu python scripts/report_exercise06.py
uv run --extra cpu streamlit run app/streamlit_app.py
```

La ruta corta evita el límite de DLL de Windows. Seleccionar ese mismo Python en VS Code. En otras máquinas se puede omitir UV_PROJECT_ENVIRONMENT y usar .venv dentro del proyecto. GPU es opcional con WSL2/Linux ya configurado y `--extra gpu` en lugar de cpu. No mezclar ambos extras ni instalaciones globales. El equipo usado detectó {len(meta["gpu_visible"])} GPU en TensorFlow. El entrenamiento usa CPU y los pesos se entregan en models/.

La ejecución completa tarda aproximadamente {meta["total_seconds"] / 60:.1f} min en este equipo; varía con carga. El notebook vuelve a entrenar. Los scripts originales del profesor permanecen en el historial; la entrega completa se ejecuta con run_exercise06.py. La app solo hace inferencia y puede utilizar los pesos incluidos sin volver a entrenar.

## Datos y metodología
Fuente: [Fashion-MNIST oficial](https://github.com/zalandoresearch/fashion-mnist), licencia MIT. Los4 archivos IDX comprimidos se descargan si faltan y se verifican con SHA-256 completo en docs/data_source.json. Los datos originales no se versionan. No se usa MovieLens en esta práctica.

60,000 imágenes de desarrollo separadas por permutación semilla 42 en 54,000/6,000; 10,000 prueba oficial. Índices guardados en reports/split_indices.npz. Etiquetas solo para errores por clase, nunca como entrada ni objetivo de los modelos. Normalización float32/255. AE 16,6 épocas; VAE 2,8 épocas por beta; lote 256. Checkpoint AE: menor BCE de validación. Checkpoint y beta VAE: menor R+KL de validación con muestreo fijo. Una semilla de ajuste limita generalización estadística.

Panel16 con semilla 2026;256 muestras del prior normal con semilla 31415;256 casos reales de validación con semilla 2718. Vecinos exactos contra54,000 imágenes de ajuste. Métricas de píxeles y paneles no son FID, no prueban originalidad ni privacidad. Ver Model Card para límites completos.

## Verificación y estructura
13 pruebas: contratos de forma, normalización, partición, MSE, KL analítica, reducción de pérdidas, gradientes de reparametrización, interpolación y vecinos. Notebook ejecutado de principio a fin. src/ contiene modelos y evaluación; tests/ pruebas; scripts/ ejecución y PDF; reports/ evidencias; models/ pesos Keras; app/ Streamlit. uv.lock y requirements.txt fijan dependencias.

## Fuentes y asistencia
[Mandato](https://uasdvirtual.uasd.edu.do/diplomados/mod/assign/view.php?id=238699), [LAB10](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=238956&chapterid=784), [aclaraciones AE/VAE](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974&chapterid=847), [calidad perceptual](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974&chapterid=850).

Codex (OpenAI) apoyó implementación, comprobaciones y redacción. Conclusión como borrador para revisión del estudiante; no se afirma que la revisión personal ya se realizó. Detalles en AI_USE_DECLARATION.md.
""",
        encoding="utf-8",
    )
    (ROOT / "MODEL_CARD.md").write_text(
        f"""# Model Card — AE/VAE de Fashion-MNIST

Autor: Alexis De La Cruz (100805370). Base académica: Edwin Ramón José Nolasco. Uso previsto: demostración educativa local de reconstrucción y espacio latente. No es una herramienta de producción, identificación de personas ni un mecanismo de anonimización.

## Arquitectura y ajuste
AE: Flatten 784 → Dense16 ReLU → Dense784 sigmoid →28x28x1. VAE: Flatten 784 → Dense128 ReLU → medias y logvar de2 dimensiones; decoder2 → Dense128 ReLU → Dense784 sigmoid →28x28x1. Adam con learning rate0.001; lote 256. AE6 épocas, VAE8 por beta1/0.1, sin warm-up. Semilla 42. Las arquitecturas se preservan del ZIP; capacidad desigual impide aislar el efecto de KL.

R=media por lote de la suma BCE en 784 píxeles. KL=media de 0.5*sum(mu²+exp(logvar)−1−logvar). Entrenamiento VAE: R+beta*KL. Evaluación de validación con epsilon normal stateless y fijo por lote, para comparar checkpoints; regla común R+KL con peso1. Se seleccionó {NAMES[choice]}, época {int(selected.best_epoch)}. Decisión registrada en reports/decision_locked.json antes de prueba. No se eligieron imágenes a mano.

## Datos
Fashion-MNIST, fuente oficial Zalando Research, MIT. 60,000 imágenes originales para54,000 ajuste y 6,000 validación por permutación; 10,000 prueba oficial. Float32 en[0,1],28x28x1. Etiquetas solo para análisis de errores. SHA-256 de fuentes en docs/data_source.json; huella de partición {meta["split_sha256"]}. No se utiliza test para checkpoints/beta.

## Resultados y costo
{table}

Prueba VAE reconstruida desde mu; KL calculada con mu/logvar. SSIM con tf.image.ssim,max_val1 (ventana por defecto11). Latencia mediana de3 repeticiones después de calentamiento, lote 256, ms por imagen; no percentil95 ni SLA. Tiempo de ajuste incluye validación y guardado de checkpoints. Tamaño VAE incluye encoder+decoder; AE corresponde a autoencoder.keras. Pico working set del proceso: {meta["peak_working_set_mib"]:.1f} MiB. Duración total {meta["total_seconds"]:.1f}s. Python {meta["python"]},TensorFlow {meta["tensorflow"]},Windows 11,Intel i9-13980HX,RAM 16 GB,NVIDIA RTX4070Laptop8GB; GPU visible en TensorFlow: {meta["gpu_visible"]}. Costos dependen de carga/hardware.

## Generación, diversidad y memoria
256 vectores normales N(0,I), semilla 31415. Panel16 original elegido semilla 2026. Interpolación lineal entre medias de los primeros2 originales del panel; no implica ejes semánticos. Diversidad MSE media entre todos los pares: generado {div["generated_pairwise_mse"]:.6f},real validación {div["real_validation_pairwise_mse"]:.6f}. Referencia real256,semilla 2718. No mide clases cubiertas ni calidad humana.

Vecinos exactos en píxeles contra todo el ajuste: mediana generada {div["generated_nn_mse_median"]:.6f};mediana validación-real {div["validation_nn_mse_median"]:.6f}. Umbral exploratorio p05 de referencia real {div["threshold_validation_nn_p05"]:.6f};fracción generada por debajo {div["fraction_generated_below_threshold"]:.1%}. Esa proximidad también puede deberse a salidas promedio; no demuestra copia. Los paneles muestran las primeras8 muestras sin ordenar por distancia. No se ejecutó una prueba formal de inferencia de pertenencia, FID ni evaluación humana. No afirmar privacidad u originalidad.

## Límites y decisión
Solo una semilla de ajuste; presupuesto corto; imágenes pequeñas de prendas y contexto distinto de fotografía real. El número de dimensiones con Var(mu)>0.01 es diagnóstico heurístico, no prueba de ausencia de colapso. La KL puede diferir entre objetivos; cero aislado no basta para diagnosticar posterior collapse. AE se conserva como referencia de reconstrucción y {NAMES[choice]} alimenta la app de muestreo. No se entrenó GAN: usa adversarios y puede colapsar modos; transferencia reutiliza representaciones y no es por sí sola generación. Verificación: 13 pruebas y notebook; apoyo de IA declarado.
""",
        encoding="utf-8",
    )
    # Misma identidad visual del informe CNN/Ejercicio05: carta, Calibri, navy, ficha, tablas.
    navy = colors.HexColor("#12355B")
    border = colors.HexColor("#C9CED6")
    zebra = colors.HexColor("#F2F4F7")
    body = ParagraphStyle(
        "body",
        fontName="Body",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
        textColor=colors.HexColor("#1f1f1f"),
    )
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11, spaceAfter=5)
    hs = ParagraphStyle(
        "h",
        fontName="Body-Bold",
        fontSize=12.5,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=navy,
    )
    story = []

    def p(x, sm=False):
        story.append(Paragraph(x, small if sm else body))

    def h(x):
        story.append(Paragraph(x, hs))

    def table(rows, widths, meta=False):
        data = []
        for i, row in enumerate(rows):
            cells = []
            for j, val in enumerate(row):
                st = ParagraphStyle(
                    "cell",
                    parent=small,
                    spaceAfter=0,
                    fontName="Body-Bold"
                    if (i == 0 and not meta) or (meta and j == 0)
                    else "Body",
                    textColor=colors.white
                    if i == 0 and not meta
                    else colors.HexColor("#1f1f1f"),
                )
                cells.append(Paragraph(escape(str(val)), st))
            data.append(cells)
        tb = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=0 if meta else 1)
        cmds = [
            ("GRID", (0, 0), (-1, -1), 0.35, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if meta:
            cmds.append(("BACKGROUND", (0, 0), (0, -1), zebra))
        else:
            cmds.append(("BACKGROUND", (0, 0), (-1, 0), navy))
            for i in range(2, len(rows), 2):
                cmds.append(("BACKGROUND", (0, i), (-1, i), zebra))
        tb.setStyle(TableStyle(cmds))
        story.append(tb)
        story.append(Spacer(1, 6))

    def fig(name, width=486):
        path = out / "figures" / name
        w, hh = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=width, height=width * hh / w))
        story.append(Spacer(1, 4))

    banner = Table(
        [
            [
                Paragraph(
                    "UNIVERSIDAD AUTÓNOMA DE SANTO DOMINGO · MAESTRÍA EN CIENCIAS DE DATOS E INTELIGENCIA ARTIFICIAL",
                    ParagraphStyle(
                        "bsmall",
                        fontName="Body-Bold",
                        fontSize=7,
                        leading=9,
                        textColor=colors.HexColor("#CAD8E8"),
                    ),
                )
            ],
            [
                Paragraph(
                    "Autoencoder y VAE con Fashion-MNIST",
                    ParagraphStyle(
                        "bt",
                        fontName="Body-Bold",
                        fontSize=19,
                        leading=23,
                        textColor=colors.white,
                    ),
                )
            ],
            [
                Paragraph(
                    "Mandato · Ejercicio 06 · Informe de entrega",
                    ParagraphStyle(
                        "bs",
                        fontName="Body",
                        fontSize=10.5,
                        leading=14,
                        textColor=colors.white,
                    ),
                )
            ],
        ],
        colWidths=[486],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), navy),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 16))
    table(
        [
            ["Asignatura", "INF-8239 Ciencia de Datos II"],
            ["Unidad", "03 · Algoritmos y Características Generativas"],
            ["Código", "U03.E06"],
            ["Estudiante", "Alexis De La Cruz · Matrícula 100805370"],
            ["Fecha", "19 de octubre de 2026"],
            [
                "Entorno",
                f"Windows 11 · Intel i9-13980HX · RAM 16 GB · NVIDIA RTX 4070 Laptop, 8 GB · Python {meta['python']} · TensorFlow {meta['tensorflow']}",
            ],
        ],
        [100, 386],
        True,
    )
    h("1. Repositorio y propósito")
    p(f'<b>Enlace:</b> <link href="{URL}" color="#1F5FAF"><u>{URL}</u></link>')
    p(
        "Comparar reconstrucción y generación sobre el proyecto base LAB10 de Edwin Ramón José Nolasco. Se entregan notebook ejecutado, modelos, pruebas, aplicación local y Model Card. El AE comprime una entrada; el VAE aprende una distribución latente regularizada."
    )
    h("2. Datos y protocolo")
    p(
        "Fashion-MNIST, fuente oficial Zalando Research, licencia MIT. 60,000 imágenes de desarrollo se separan en 54,000 ajuste y 6,000 validación; 10,000 de prueba oficial. Normalización float32/255; archivos verificados con SHA-256. Etiquetas reservadas para análisis por clase."
    )
    p(
        "Semilla 42, lote 256. AE 16: 6 épocas; VAE 2: 8 épocas con β=1 y β=0.1. Se preservan las arquitecturas de la base; capacidad y tamaño latente distintos impiden aislar solo el efecto de KL. Los 16 casos de prueba y las 256 muestras nuevas se fijan con semillas, sin escoger los mejores."
    )
    h("3. Resultados y elección previa a prueba")
    rows = [["Modelo", "MSE prueba", "SSIM", "KL prueba", "Época elegida"]]
    for key, r in t.iterrows():
        rows.append(
            [
                NAMES[key],
                f"{r.test_mse:.5f}",
                f"{r.test_ssim:.4f}",
                "—" if pd.isna(r.test_kl) else f"{r.test_kl:.3f}",
                int(r.best_epoch),
            ]
        )
    table(rows, [145, 90, 70, 85, 96])
    p(
        f"Selección: <b>{NAMES[choice]}</b>, por menor reconstrucción sumada + KL de validación con peso común 1. La decisión se registró antes de prueba. AE usa menor BCE de validación. VAE reconstruye desde μ para comparar pares de forma determinista.",
        True,
    )
    story.append(PageBreak())
    h("4. Panel fijo: reconstrucción y generación")
    fig("panel16.png", 480)
    p(
        "Figura 1. Dieciséis casos de prueba seleccionados con semilla 2026. Cada bloque presenta originales, AE, VAE desdeμ y muestras nuevas del prior N(0,I), semilla 31415. La última fila es generación libre: no corresponde a la original de la columna.",
        True,
    )
    p(
        "MSE y SSIM describen similitud con una referencia emparejada. No demuestran calidad de las muestras nuevas. El panel permite observar pérdida de detalles, siluetas ambiguas y variación; su selección se conserva en visual_evidence.npz.",
        True,
    )
    errors=pd.read_csv(out/'errors_by_class.csv')
    names=['Camiseta','Pantalón','Jersey','Vestido','Abrigo','Sandalia','Camisa','Zapatilla','Bolso','Botín']
    worst=errors.loc[errors.model==choice].sort_values('mse',ascending=False).head(3)
    table([['Clases con mayor error del VAE elegido','MSE','Casos']]+[[names[int(r['class'])],f"{r['mse']:.5f}",int(r['n'])] for _,r in worst.iterrows()],[326,80,80])
    p('El bolso del caso 797 pierde sus asas y se aproxima a una prenda superior; los estampados finos del caso 7211 se diluyen. Son fallos visibles del panel fijado, no ejemplos añadidos después de revisar resultados.',True)
    story.append(PageBreak())
    h("5. Curvas, espacio latente y vecinos")
    fig("curves.png")
    p(
        "Figura 2. Reconstrucción y KL separadas. En VAE, línea continua=validación y discontinua=ajuste. AE promedia BCE por píxel; VAE la suma en 784 píxeles y KL en 2 dimensiones, con promedio por imagen. La escala de sus pérdidas no se compara directamente.",
        True,
    )
    fig("interpolation.png")
    p(
        "Figura 3. Interpolación entre las medias latentes de los primeros dos casos del panel. La continuidad no implica ejes con significado semántico garantizado.",
        True,
    )
    fig("neighbors.png")
    p(
        "Figura 4. Primeras ocho muestras del prior y vecino exacto en píxeles entre 54,000 imágenes de ajuste. No se ordenaron por parecido.",
        True,
    )
    table(
        [
            ["Diagnóstico sobre 256 muestras", "Generadas", "Referencia real"],
            [
                "Diversidad media entre pares (MSE)",
                f"{div['generated_pairwise_mse']:.5f}",
                f"{div['real_validation_pairwise_mse']:.5f}",
            ],
            [
                "Mediana MSE al vecino de entrenamiento",
                f"{div['generated_nn_mse_median']:.5f}",
                f"{div['validation_nn_mse_median']:.5f}",
            ],
        ],
        [286, 100, 100],
    )
    p(
        "Referencia real: 256 casos de validación, semilla 2718. La proximidad puede reflejar una forma promedio; no demuestra copia. Estas medidas no son FID ni una garantía de privacidad, diversidad de clases u originalidad.",
        True,
    )
    story.append(PageBreak())
    h("6. Costo y conclusión")
    rows = [["Modelo", "Ajuste (s)", "Inferencia (ms/img)", "Tamaño (KiB)"]]
    for key, r in t.iterrows():
        rows.append(
            [NAMES[key], f"{r.train_s:.1f}", f"{r.latency_ms:.3f}", f"{r.size_kib:.1f}"]
        )
    table(rows, [145, 90, 135, 116])
    p(
        f"Latencia mediana tras calentamiento, lote 256. Tamaño VAE incluye encoder+decoder. Pico working set del proceso: {meta['peak_working_set_mib']:.1f} MiB. Costos dependientes del equipo y su carga.",
        True,
    )
    for para in conclusion.split("\n\n")[1:]:
        p(escape(para))
    h("7. Referencias y nota del autor")
    p(
        'Kingma, D. P., &amp; Welling, M. (2013). <i>Auto-Encoding Variational Bayes</i>. <link href="https://doi.org/10.48550/arXiv.1312.6114">doi:10.48550/arXiv.1312.6114</link>. Zalando Research. (s. f.). <i>Fashion-MNIST</i> [Dataset, MIT]. github.com/zalandoresearch/fashion-mnist. Nolasco, E. R. J. (s. f.). <i>LAB10 y aclaraciones U03</i> [Material de curso]. UASD.',
        True,
    )
    p(
        "Codex (OpenAI) apoyó código, verificación y redacción. Procedencia, prompts y correcciones se documentan en AI_USE_DECLARATION.md. Conclusión como borrador para revisión personal del estudiante.",
        True,
    )

    class Numbered(Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.states = []

        def showPage(self):
            self.states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self.states)
            for s in self.states:
                self.__dict__.update(s)
                self.setFont("Body", 8)
                self.setFillColor(colors.HexColor("#5F5E5A"))
                self.drawRightString(
                    549,
                    34,
                    f"U03.E06 · Alexis De La Cruz · página {self._pageNumber} de {n}",
                )
                super().showPage()
            super().save()

    pdf = out / "informe_ejercicio06.pdf"
    SimpleDocTemplate(
        str(pdf),
        pagesize=(612, 792),
        leftMargin=57,
        rightMargin=57,
        topMargin=56,
        bottomMargin=50,
        title="U03.E06 · AE y VAE · Alexis De La Cruz",
        author="Alexis De La Cruz",
    ).build(story, canvasmaker=Numbered)
    dest = ROOT.parent / "Unidad03-Ejercicio06-Generativos"
    dest.mkdir(exist_ok=True)
    shutil.copy2(pdf, dest / "100805370_DeLaCruz_U03E06_Generativos.pdf")
    print("Informe generado:", pdf)


if __name__ == "__main__":
    main()
