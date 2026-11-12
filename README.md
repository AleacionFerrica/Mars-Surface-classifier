# MarsProject: clasificación de escenas marcianas con pocas muestras (SE-ResNet-MTL)

Proyecto personal de visión por computador sobre imágenes del rover Curiosity (MSL). El objetivo es **replicar en local y con mi propio dispositivo la metodología del artículo**:

> He, Y.; Shen, N.; Zhang, X.; Wu, Z.; Li, J.; Hou, D. *Research on Few-Shot Mars Rover Onboard Surface Scene Classification Based on SE-ResNet-MTL*. Remote Sensing 2026, 18, 2319. https://doi.org/10.3390/rs18142319
>; los autores publicaron código y dataset en Zenodo: https://zenodo.org/records/20828875)

> **Estado:** fase inicial. El dataset del artículo ya está en `data/`. Todavía no hay código: los modelos se implementarán desde cero, sin reutilizar el código publicado por los autores.

---

## 1. Hardware

| Componente | Este proyecto | Artículo |
|---|---|---|
| GPU | NVIDIA RTX 3060, **12 GB** VRAM | NVIDIA RTX 4060, 8 GB VRAM |
| CPU | Intel Core i7-12700F | AMD Ryzen 9 7945HX |
| RAM | 16 GB | no indicada |
| SO / Python | Windows 11, Python 3.12 | no indicado |

### Entorno: WSL2

En Windows nativo hay limitaciones con CUDA (y con `torch.compile` y otras herramientas), así que **todo lo que use la GPU se ejecuta en WSL2 (Ubuntu)**. 

```bash
# Dentro de WSL (Ubuntu). 
sudo apt install python3-venv        
python3 -m venv ~/venvs/mars
source ~/venvs/mars/bin/activate
pip install torch torchvision          
pip install numpy pandas matplotlib seaborn pillow scikit-learn imagehash jupyter ipykernel
python -m ipykernel install --user --name mars --display-name "Python (mars, WSL)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La 3060 tiene más VRAM que la GPU del artículo, la configuración original (batch 8, ResNet50 a 224×224). Lo que más memoria consume es MAML: conserva el grafo completo del bucle interno a través de ResNet50 y procesa en cada tarea 5×(K+15) imágenes, es decir, 80 en 1-shot y 100 en 5-shot.

---

## 2. Datos

### 2.1 Dataset disponible en local: `msl-images/`

Es el dataset *MSL Curiosity labeled data set* de Stanboli & Wagstaff (DOI [10.5281/zenodo.1049137](https://doi.org/10.5281/zenodo.1049137)), el que se usó en *Deep Mars* (Wagstaff et al., AAAI 2018).

- 6691 imágenes etiquetadas de unos 256×256 px en versión *browse* (`calibrated/` tiene 6737 ficheros, 46 de ellos sin etiqueta).
- Instrumentos, según el código del nombre de fichero: Mastcam Left `ML` (1882), Mastcam Right `MR` (2402) y MAHLI `MH` (2407).
- **25 clases** de contenido (`msl_synset_words-indexed.txt`): partes del rover (`wheel`, `drill`, `scoop`, `turret`…), objetivos de calibración, `ground`, `horizon`, `sun`, etc.
- Particiones **por sol**, en orden cronológico:

| Split | Imágenes | Soles |
|---|---|---|
| train | 3746 | 3–180 |
| val | 1640 | 182–564 |
| test | 1305 | 565–1060 |

Los ficheros de split tienen el formato `calibrated/<archivo>.JPG <id_clase>`. Los cuatro primeros dígitos del nombre de archivo son el sol.

### 2.2 Dataset del artículo

El artículo construye un dataset propio de **1000 imágenes RGB de Mastcam** (a partir de MSL y MSL-v2) con **5 clases de escena**:

| Clase | Descripción | Muestras |
|---|---|---|
| Artifacts | Componentes del rover o del módulo de aterrizaje | 193 |
| Invalid | Imágenes borrosas, muy oscuras, de cielo o vistas lejanas irrelevantes | 176 |
| Regolith | Arena, suelo, grava y sedimento suelto | 227 |
| Rocks | Rocas bien delimitadas | 223 |
| Rover_tracks | Huellas de las ruedas | 181 |

- Se guardan como máximo 2 imágenes por clase y sol, para evitar casi-duplicados.
- Si una imagen encaja en varias clases, manda esta prioridad: `Invalid > Artifacts > Rover_tracks > Rocks > Regolith`.
- Según el texto, las imágenes se ordenan por sol: el último 20 % de los soles forma validación y el 80 % anterior, entrenamiento (división 8:2 sin soles compartidos).

| Clase | train | val |
|---|---|---|
| artifacts | 155 | 38 |
| invalid | 141 | 35 |
| regolith | 182 | 45 |
| rocks | 178 | 45 |
| rover_tracks | 143 | 38 |

- Todas las imágenes son JPG RGB de 560×500 px, de Mastcam Left (528) y Right (472).
- El sol está en el nombre de archivo, justo antes de `ML`/`MR`. Por ejemplo, `1000_1000MR0044630120503572E01_DXXX.jpg` es del sol 1000. Casi siempre son 4 dígitos con ceros a la izquierda, pero **hay 2 archivos con 3** (`cr_492_492ML…`, en `train/invalid`). Algunos nombres llevan el prefijo `cr_`.

### 2.4 Decisión

Se trabaja con el dataset de 5 clases de `data/` para que los resultados sean comparables con el artículo. 

**Partición propia train / val / test (más realista que la del artículo).** El artículo solo tiene train y val, y usa val tanto para elegir el checkpoint como para informar los resultados, lo que da resultados optimistas. Este proyecto:
- junta las 1000 imágenes y crea **tres splits sin soles compartidos**, en orden cronológico si el reparto de clases lo permite (en otro caso, con `StratifiedGroupKFold` agrupando por sol);
- usa val para early stopping y para elegir hiperparámetros, y deja **test intacto hasta la evaluación final**;
- exige que cada clase tenga **≥ 20 imágenes** en los splits donde se muestrean episodios 5-way 5-shot (5 de soporte + 15 de query);
- guarda la partición en `splits/split_v1.csv` para que sea reproducible.

El diseño concreto se decide en la sección 9 de `experiments.ipynb`. Para comparar con las tablas del artículo se puede repetir el entrenamiento con la partición original.

---

## 3. Metodología a replicar

Se parte de ResNet50 y se le añaden mejoras una a una (ablación):

1. **Baseline:** ResNet50 entrenada desde cero, con una capa de salida de 5 clases.
2. **Transfer learning:** pesos de ImageNet, backbone **completamente congelado**; solo se entrena la cabeza de clasificación.
3. **SE-ResNet50:** un bloque Squeeze-and-Excitation (reducción r = 16) dentro de cada bottleneck, antes de la suma residual. Se comparan las activaciones ReLU, GELU, HardSwish y LeakyReLU; el artículo elige **ReLU**.
4. **SE-ResNet-MTL:** meta-entrenamiento MAML sobre el modelo SE-ResNet50 (configuración en la tabla de abajo).

### Preprocesado
- Redimensionado a 224×224 y normalización con la media y desviación de ImageNet (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
- Aumentación online, solo en entrenamiento: volteo horizontal, brillo y contraste. La Fig. 2 del artículo muestra además recorte central y ruido gaussiano.

### Hiperparámetros del entrenamiento supervisado
| Parámetro | Valor |
|---|---|
| Optimizador / pérdida | Adam / entropía cruzada |
| Batch size | 8 |
| LR inicial / weight decay | 1e-3 / 1e-4 |
| Scheduler | ReduceLROnPlateau (×0.5, paciencia 5, sobre val acc) |
| Early stopping | paciencia 20, máximo 200 épocas |
| Checkpoint final | el de menor pérdida de validación |

### Hiperparámetros de MAML
| Parámetro | Valor |
|---|---|
| Configuración | 5-way, K = 1 y K = 5, 15 imágenes de query por clase |
| Tareas por época | 20 |
| LR interno (SGD) / externo (Adam) | 0.01 / 0.001 (weight decay 1e-4) |
| Gradientes | se conserva el grafo completo del bucle interno (sin aproximación de primer orden) |
| Early stopping | sobre la meta-pérdida, paciencia 5, máximo 40 meta-épocas |
| Evaluación | media e IC 95 % sobre 10 episodios aleatorios |

### Reproducibilidad (como en el artículo)
- Semilla 42 en `torch`, `numpy`, `random` y CUDA; `torch.backends.cudnn.deterministic = True`.
- TF32 desactivado y `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Figuras exportadas a 600 DPI.

### Métricas
Accuracy, Precision, Recall y F1 (macro); **Acc Gap** = Acc_train − Acc_val; matrices de confusión; accuracy 1-shot/5-shot con IC 95 %. Para comparar activaciones se mide también el tiempo de inferencia (ms), los FLOPs, los parámetros y la época de convergencia.

### Comparativas adicionales del artículo
- **Otros métodos de meta-aprendizaje** sobre el mismo backbone SE-ResNet50: FOMAML, Reptile y Meta-Baseline.
- **Backbones ligeros** con fine-tuning estándar: ResNet18, MobileNetV2, EfficientNet-B0 y ShuffleNetV2.

---

## 4. Resultados de referencia (artículo)

| Modelo | Acc (%) | F1 (%) | 1-shot | 5-shot |
|---|---|---|---|---|
| ResNet50 baseline | 78.11 | 77.54 | 0.573 ± 0.069 | 0.685 ± 0.036 |
| ResNet50 + TL | 93.53 | 93.82 | 0.445 ± 0.082 | 0.733 ± 0.024 |
| SE-ResNet50 + TL | 95.52 | 95.66 | 0.477 ± 0.061 | 0.661 ± 0.045 |
| ResNet-MTL | 86.07 | 86.46 | 0.873 ± 0.037 | 0.863 ± 0.017 |
| **SE-ResNet-MTL** | 90.05 | 90.01 | **0.905 ± 0.014** | **0.915 ± 0.026** |

Meta-aprendizaje (1-shot / 5-shot): FOMAML 0.898 / 0.892, Reptile 0.855 / 0.872, Meta-Baseline 0.498 / 0.560.

---

## 5. Puntos ambiguos del artículo a resolver al replicar

- **Pesos del SE con el backbone congelado:** el artículo dice que los módulos SE quedan fijos si el backbone está congelado. Si se insertan en una ResNet50 de ImageNet, esos pesos serían aleatorios. Lo razonable es partir de una SE-ResNet50 ya preentrenada (p. ej. `seresnet50` de `timm`) o dejar los SE entrenables. Hay que documentar la opción elegida.
- **Qué parámetros adapta MAML** en el bucle interno (toda la red o solo la cabeza): no se especifica, y afecta mucho a la memoria y al tiempo de entrenamiento.
- **"Sin derivadas de segundo orden explícitas" pero "conservando el grafo completo":** la descripción es contradictoria. FOMAML aparece como método distinto, así que lo más probable es que se refiera a MAML de segundo orden (`create_graph=True`).

---

## 6. Estructura actual del repositorio

```
MarsProject/
├── data/                       # Dataset del artículo (5 clases, train/val), solo datos   [no incluido: Zenodo]
├── msl-images/                 # Dataset MSL original (25 clases)                       [no incluido: Zenodo]
├── remotesensing-18-02319.pdf  # Artículo de referencia                                 [no incluido: ver §7]
├── experiments.ipynb           # EDA guiado y diseño de la partición
├── 00_practica_pytorch.ipynb   # 6 ejercicios guiados con autocomprobación (núcleo de PyTorch)
├── 01_cnn_baseline.ipynb       # CNN propia desde cero + medición del entrenamiento en GPU
├── splits/split_v1.csv         # Partición propia train/val/test (se genera en experiments.ipynb §9)
├── results/<run>/              # history.csv, curvas, summary.json de cada experimento (best.pt no se sube)
└── README.md
```

## 7. Citas

- Stanboli, A.; Wagstaff, K. *Mars surface image (Curiosity rover) labeled data set*. Zenodo. DOI 10.5281/zenodo.1049137
- Wagstaff, K. et al. *Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging Atlas*. AAAI/IAAI 2018.
- He, Y. et al. *Research on Few-Shot Mars Rover Onboard Surface Scene Classification Based on SE-ResNet-MTL*. Remote Sensing 2026, 18, 2319.
