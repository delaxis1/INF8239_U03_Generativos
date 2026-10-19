# Model Card — AE/VAE de Fashion-MNIST

Autor: Alexis De La Cruz (100805370). Base académica: Edwin Ramón José Nolasco. Uso previsto: demostración educativa local de reconstrucción y espacio latente. No es una herramienta de producción, identificación de personas ni un mecanismo de anonimización.

## Arquitectura y ajuste
AE: Flatten 784 → Dense16 ReLU → Dense784 sigmoid →28x28x1. VAE: Flatten 784 → Dense128 ReLU → medias y logvar de2 dimensiones; decoder2 → Dense128 ReLU → Dense784 sigmoid →28x28x1. Adam con learning rate0.001; lote 256. AE6 épocas, VAE8 por beta1/0.1, sin warm-up. Semilla 42. Las arquitecturas se preservan del ZIP; capacidad desigual impide aislar el efecto de KL.

R=media por lote de la suma BCE en 784 píxeles. KL=media de 0.5*sum(mu²+exp(logvar)−1−logvar). Entrenamiento VAE: R+beta*KL. Evaluación de validación con epsilon normal stateless y fijo por lote, para comparar checkpoints; regla común R+KL con peso1. Se seleccionó VAE · β=1, época 8. Decisión registrada en reports/decision_locked.json antes de prueba. No se eligieron imágenes a mano.

## Datos
Fashion-MNIST, fuente oficial Zalando Research, MIT. 60,000 imágenes originales para54,000 ajuste y 6,000 validación por permutación; 10,000 prueba oficial. Float32 en[0,1],28x28x1. Etiquetas solo para análisis de errores. SHA-256 de fuentes en docs/data_source.json; huella de partición e1d0864b7fb496c54c49591955b65c82205638333445478817bc477a642b00e0. No se utiliza test para checkpoints/beta.

## Resultados y costo
| model           |   test_mse |   test_ssim |   test_kl |   parameters |   train_s |   latency_ms |   size_kib |
|:----------------|-----------:|------------:|----------:|-------------:|----------:|-------------:|-----------:|
| AE · latente 16 |    0.02040 |     0.53654 | nan       |  25888.00000 |   2.97179 |      0.00819 |  122.54297 |
| VAE · β=1       |    0.03361 |     0.41756 |   6.27843 | 202516.00000 |   7.57207 |      0.01696 |  832.97070 |
| VAE · β=0.1     |    0.03134 |     0.43879 |  19.29007 | 202516.00000 |   7.40748 |      0.01556 |  832.97070 |

Prueba VAE reconstruida desde mu; KL calculada con mu/logvar. SSIM con tf.image.ssim,max_val1 (ventana por defecto11). Latencia mediana de3 repeticiones después de calentamiento, lote 256, ms por imagen; no percentil95 ni SLA. Tiempo de ajuste incluye validación y guardado de checkpoints. Tamaño VAE incluye encoder+decoder; AE corresponde a autoencoder.keras. Pico working set del proceso: 1180.0 MiB. Duración total 21.8s. Python 3.12.12,TensorFlow 2.21.0,Windows 11,Intel i9-13980HX,RAM 16 GB,NVIDIA RTX4070Laptop8GB; GPU visible en TensorFlow: []. Costos dependen de carga/hardware.

## Generación, diversidad y memoria
256 vectores normales N(0,I), semilla 31415. Panel16 original elegido semilla 2026. Interpolación lineal entre medias de los primeros2 originales del panel; no implica ejes semánticos. Diversidad MSE media entre todos los pares: generado 0.067258,real validación 0.171946. Referencia real256,semilla 2718. No mide clases cubiertas ni calidad humana.

Vecinos exactos en píxeles contra todo el ajuste: mediana generada 0.008960;mediana validación-real 0.014499. Umbral exploratorio p05 de referencia real 0.004642;fracción generada por debajo 9.0%. Esa proximidad también puede deberse a salidas promedio; no demuestra copia. Los paneles muestran las primeras8 muestras sin ordenar por distancia. No se ejecutó una prueba formal de inferencia de pertenencia, FID ni evaluación humana. No afirmar privacidad u originalidad.

## Límites y decisión
Solo una semilla de ajuste; presupuesto corto; imágenes pequeñas de prendas y contexto distinto de fotografía real. El número de dimensiones con Var(mu)>0.01 es diagnóstico heurístico, no prueba de ausencia de colapso. La KL puede diferir entre objetivos; cero aislado no basta para diagnosticar posterior collapse. AE se conserva como referencia de reconstrucción y VAE · β=1 alimenta la app de muestreo. No se entrenó GAN: usa adversarios y puede colapsar modos; transferencia reutiliza representaciones y no es por sí sola generación. Verificación: 13 pruebas y notebook; apoyo de IA declarado.
