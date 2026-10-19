# INF8239_U03_Generativos

Alexis De La Cruz · 100805370 · INF-8239 · Ejercicio 06.

Base oficial LAB10 de Edwin Ramón José Nolasco (commit acfd6b9). Autoencoder y VAE con Fashion-MNIST completo. [Notebook ejecutado](notebooks/06_ae_vae.ipynb), [informe PDF](reports/informe_ejercicio06.pdf), [Model Card](MODEL_CARD.md), [declaración IA](AI_USE_DECLARATION.md).

## Resultados
Selección de generación en validación: **VAE · β=1**. Prueba: 10,000 imágenes; reconstrucción VAE desde la media latente.

| model           |   test_mse |   test_ssim |   test_kl |   parameters |   train_s |   latency_ms |   size_kib |
|:----------------|-----------:|------------:|----------:|-------------:|----------:|-------------:|-----------:|
| AE · latente 16 |    0.02040 |     0.53654 | nan       |  25888.00000 |   2.97179 |      0.00819 |  122.54297 |
| VAE · β=1       |    0.03361 |     0.41756 |   6.27843 | 202516.00000 |   7.57207 |      0.01696 |  832.97070 |
| VAE · β=0.1     |    0.03134 |     0.43879 |  19.29007 | 202516.00000 |   7.40748 |      0.01556 |  832.97070 |

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

La ruta corta evita el límite de DLL de Windows. Seleccionar ese mismo Python en VS Code. En otras máquinas se puede omitir UV_PROJECT_ENVIRONMENT y usar .venv dentro del proyecto. GPU es opcional con WSL2/Linux ya configurado y `--extra gpu` en lugar de cpu. No mezclar ambos extras ni instalaciones globales. El equipo usado detectó 0 GPU en TensorFlow. El entrenamiento usa CPU y los pesos se entregan en models/.

La ejecución completa tarda aproximadamente 0.4 min en este equipo; varía con carga. El notebook vuelve a entrenar. Los scripts originales del profesor permanecen en el historial; la entrega completa se ejecuta con run_exercise06.py. La app solo hace inferencia y puede utilizar los pesos incluidos sin volver a entrenar.

## Datos y metodología
Fuente: [Fashion-MNIST oficial](https://github.com/zalandoresearch/fashion-mnist), licencia MIT. Los4 archivos IDX comprimidos se descargan si faltan y se verifican con SHA-256 completo en docs/data_source.json. Los datos originales no se versionan. No se usa MovieLens en esta práctica.

60,000 imágenes de desarrollo separadas por permutación semilla 42 en 54,000/6,000; 10,000 prueba oficial. Índices guardados en reports/split_indices.npz. Etiquetas solo para errores por clase, nunca como entrada ni objetivo de los modelos. Normalización float32/255. AE 16,6 épocas; VAE 2,8 épocas por beta; lote 256. Checkpoint AE: menor BCE de validación. Checkpoint y beta VAE: menor R+KL de validación con muestreo fijo. Una semilla de ajuste limita generalización estadística.

Panel16 con semilla 2026;256 muestras del prior normal con semilla 31415;256 casos reales de validación con semilla 2718. Vecinos exactos contra54,000 imágenes de ajuste. Métricas de píxeles y paneles no son FID, no prueban originalidad ni privacidad. Ver Model Card para límites completos.

## Verificación y estructura
13 pruebas: contratos de forma, normalización, partición, MSE, KL analítica, reducción de pérdidas, gradientes de reparametrización, interpolación y vecinos. Notebook ejecutado de principio a fin. src/ contiene modelos y evaluación; tests/ pruebas; scripts/ ejecución y PDF; reports/ evidencias; models/ pesos Keras; app/ Streamlit. uv.lock y requirements.txt fijan dependencias.

## Fuentes y asistencia
[Mandato](https://uasdvirtual.uasd.edu.do/diplomados/mod/assign/view.php?id=238699), [LAB10](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=238956&chapterid=784), [aclaraciones AE/VAE](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974&chapterid=847), [calidad perceptual](https://uasdvirtual.uasd.edu.do/diplomados/mod/book/view.php?id=239974&chapterid=850).

Codex (OpenAI) apoyó implementación, comprobaciones y redacción. Conclusión como borrador para revisión del estudiante; no se afirma que la revisión personal ya se realizó. Detalles en AI_USE_DECLARATION.md.
