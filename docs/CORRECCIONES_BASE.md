# Procedencia, cambios y protocolo de la práctica

Base oficial preservada en acfd6b9. El script original train_ae_vae.py permanece como referencia; para reproducir la entrega completa usar run_exercise06.py o el notebook06.

- Arquitecturas originales preservadas: AE16 y VAE2. Diferencias de capacidad impiden atribuir todo a KL.
- Reparametrización y objetivos pasan a src/objectives.py; se configuran semilla y determinismo antes de operar.
- Validación aleatoria fija de6,000 ejemplos para ambos;10,000 de prueba reservados. AE guarda menor BCE; VAE guarda menor R+KL común a beta1/0.1.
- Entrenamiento manual por lotes para conservar índices y semillas explícitos; Adam0.001, lote256,6/8 épocas completas. AE de la base permitía parada temprana; aquí se ejecuta el presupuesto completo y se restaura el mejor checkpoint de validación.
- BCE AE media por píxel; BCE VAE suma784; KL suma2; promedio por imagen. Evaluación VAE desde mu documentada separadamente del muestreo posterior usado al ajustar.
- Panel16 aleatorio fijo sustituye primeros12; genera256 muestras y muestra16 sin elección manual. Vecinos exactos contra54,000 casos de ajuste.
- Sin FID/LPIPS ni clasificador juez: MSE/SSIM para reconstrucción y diversidad/vecinos/paneles como diagnóstico limitado para generación. No se declara privacidad, originalidad ni calidad humana certificada.
- CPU de TensorFlow en Windows; GPU opcional según el manual. El entorno se aisló en una ruta corta.

Fuentes: mandato238699,LAB10 capítulo784,aclaraciones capítulos847 y850 del aula UASD. La mención a MovieLens en el texto general del mandato no cambia el dataset Fashion-MNIST especificado por LAB10.
