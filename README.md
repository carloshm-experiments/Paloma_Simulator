# Simulador del Proyecto PALOMA
### Parallel Logic Machine — Prototec S.L. (~1998)  ·  v2.0
### Homenaje a Paco Menéndez

> Simulador Python de la arquitectura de computación masivamente paralela **PALOMA** (*Parallel Logic Machine*), desarrollada por Prototec S.L. en España a finales de los años noventa. El proyecto proponía una alternativa radicalmente distinta a los multiprocesadores convencionales de la época mediante el concepto de **Memoria Matricial Inteligente (MMI)**: una malla de procesadores sencillos conectados únicamente a sus cuatro vecinos inmediatos, capaz de escalar de forma perfectamente lineal. El simulador modela fielmente la arquitectura con el repertorio completo de 64 instrucciones, permite lanzar algoritmos propios desde archivos JSON y acepta todos sus parámetros desde la línea de comandos.

---

## Índice

1. [Contexto histórico](#1-contexto-histórico)
2. [La arquitectura PALOMA](#2-la-arquitectura-paloma)
3. [Qué modela este simulador](#3-qué-modela-este-simulador)
4. [Estructura del proyecto](#4-estructura-del-proyecto)
5. [Instalación y requisitos](#5-instalación-y-requisitos)
6. [Uso desde línea de comandos](#6-uso-desde-línea-de-comandos)
7. [El formato JSON de algoritmos](#7-el-formato-json-de-algoritmos)
8. [Algoritmos incluidos](#8-algoritmos-incluidos)
9. [Usar la API Python directamente](#9-usar-la-api-python-directamente)
10. [Personalización y extensión](#10-personalización-y-extensión)
11. [Repertorio completo de 64 instrucciones](#11-repertorio-completo-de-64-instrucciones)
12. [Constantes arquitectónicas y sus fuentes](#12-constantes-arquitectónicas-y-sus-fuentes)
13. [Limitaciones del modelo](#13-limitaciones-del-modelo)
14. [Referencias](#14-referencias)

---

## 1. Contexto histórico

A finales de los años ochenta, Francisco Menéndez comenzó a desarrollar una arquitectura de computación paralela masiva como derivación de su proyecto fin de carrera sobre holografía generada por ordenador. Las ecuaciones holográficas requerían una potencia de cálculo numérico varias órdenes de magnitud por encima de lo que un monoprocesador podía ofrecer, pero las soluciones existentes —multiprocesadores SMP o sistemas MPP tipo Connection Machine— tenían un problema fundamental: su escalabilidad se degradaba inevitablemente al crecer la red de comunicaciones.

La idea central de PALOMA era tan sencilla como poderosa: si cada procesador solo habla con sus cuatro vecinos inmediatos, la complejidad de la red no crece al añadir más procesadores. La potencia escala de forma perfectamente lineal, y el sistema puede integrarse en un ordenador convencional como una ampliación de memoria —de ahí el nombre *Memoria Matricial Inteligente*—. Esta compatibilidad con el bus PCI estándar era clave: hacía que la MMI fuese transparente al sistema operativo y no requiriese modificar el software del host más allá del kernel del algoritmo paralelo.

El proyecto fue presentado en el CDTI en 1990, se constituyó la empresa **Indestein** en 1991 con socios de la UCM y de la industria del software española, pero fue abandonado a finales de ese año por falta de financiación y la crisis económica. Prototec S.L. retomó el desarrollo en 1995 y finalizó un simulador completo en Windows 95 en 1998, momento del que datan los documentos técnicos en los que se basa este código.

En términos modernos, la arquitectura PALOMA anticipa conceptos que hoy reconocemos en las TPUs de Google (matrices de procesadores con comunicación local), en los aceleradores neuronales de IA, y en las FPGAs de alto rendimiento —todo ello concebido en 1989 y diseñado para fabricarse a una fracción del coste de los superordenadores de la época.

---

## 2. La arquitectura PALOMA

Entender qué simula este código requiere comprender los tres niveles de la arquitectura, que se corresponden con las tres clases principales del simulador.

### El Procesador Elemental (PE)

Cada PE es una unidad funcional completa de 32 bits con ALU y FPU, cuatro registros de desplazamiento serie de entrada (`REGA`, `REGB`, `REGC`, `REGD`), un registro de resultado de salida (`RESULT`) y un pequeño conjunto de registros de programa que se cargan una sola vez antes de iniciar la ejecución y permanecen fijos durante todo el cómputo. El registro más importante de entender es el **contador de habilitación** `ENACNT`: el PE solo ejecuta su instrucción cuando este contador vale cero. Programando su valor inicial (`INITCNT`) de forma escalonada en distintos PEs se implementa el *staggering* del pipeline, que permite activar secuencialmente los procesadores a medida que los datos fluyen por la malla.

La instrucción que ejecuta cada PE no cambia durante la ejecución. Lo que cambia en cada ciclo son los **datos** que llegan a través de los registros de entrada. Esto es el modelo de *dataflow*: no hay flujo de instrucciones, solo flujo de datos.

### La Memoria Matricial Inteligente (MMI)

Varios PEs se agrupan en una matriz para formar el **chip MMI**. Cada chip contiene 64 ó 256 PEs organizados en una malla cuadrada con buses de datos (32 bits) y de control (16 bits). Las entradas y salidas de los PEs periféricos se conectan directamente a las patillas del chip, permitiendo encadenar varios chips para ampliar la malla sin ningún cambio de arquitectura.

La **tarjeta MMI** es el producto comercial para el usuario final: una tarjeta de expansión PCI que contiene de 16 a 64 chips MMI más una FPGA que implementa la interfaz con el bus PCI y el intérprete de comandos. Desde el punto de vista del microprocesador host, la tarjeta se ve como una ampliación de memoria RAM.

### El ciclo de máquina

El ciclo de máquina se descompone en cuatro fases que se solapan de dos en dos. La **Fase 1** es la ejecución de instrucción: todos los PEs ejecutan simultáneamente. La **Fase 2** es la transmisión serie: los registros `RESULT` se propagan de 32 bits en 33 ciclos de reloj (32 bits + paridad). La **Fase 3** (solapada con la 2) es la escritura en RAM para instrucciones `ST`/`OUT`. La **Fase 4** (solapada con la 1) es la lectura de RAM para instrucciones `LD`/`IN`. El ciclo típico dura entre 50 y 64 ciclos de reloj a 66 MHz.

---

## 3. Qué modela este simulador

El simulador implementa un **modelo behavioral** de la arquitectura: modela correctamente el comportamiento lógico y las métricas de rendimiento, pero no simula los aspectos eléctricos ni el enrutamiento hardware cycle-accurate.

**Implementado con precisión arquitectónica** incluye la clase `PE` con todos sus registros internos, el repertorio completo de 64 instrucciones con sus latencias reales, el mecanismo de habilitación mediante `ENACNT`/`INITCNT`, la topología de malla 2D, el ciclo de máquina bifásico, las constantes físicas del chip MMI (66 MHz, 33 ciclos de transmisión, ~2 MFLOPS/PE), y el benchmark canónico del polinomio vectorial de la memoria técnica.

**Simplificado o no implementado** incluye el acceso real a RAM con arbitraje de bus por daisy chain (`BUSBUSY`/`BUSHOLD`), las células de interconexión programables (`CELLPRG`, `IOPRG`), el intérprete de comandos DMA de la FPGA, y la autodetección de posición de cada chip mediante token en la fase de RESET.

---

## 4. Estructura del proyecto

```
paloma_sim.py          Simulador principal (~1050 líneas)
README.md              Esta documentación
algorithms/            Algoritmos de ejemplo en formato JSON
  ├── rms.json
  ├── signal_power_db.json
  ├── dft_spectrum.json
  ├── fourier_synthesis.json
  ├── weighted_variance.json
  ├── hyperbolic_norm.json
  ├── log_energy.json
  └── mahalanobis.json
```

El archivo principal `paloma_sim.py` se organiza en ocho secciones: constantes arquitectónicas, repertorio de instrucciones (`Opcode` + `LATENCY`), la clase `PE`, la clase `MMI`, el generador de datos, el resolutor de fuentes de operandos, el ejecutor de algoritmos (`AlgorithmRunner`), las funciones de configuraciones históricas y escalabilidad, el dashboard visual, el algoritmo integrado (polinomio vectorial), y el CLI con `argparse`.

---

## 5. Instalación y requisitos

El simulador requiere **Python 3.10 o superior** por el uso de la sintaxis `match`/`case` introducida en esa versión. Las únicas dependencias externas son numpy y matplotlib:

```bash
pip install numpy matplotlib
```

No hay más dependencias. El código es portable y ejecutable en cualquier plataforma sin instalación adicional.

```bash
# Verificar versión
python3 --version   # debe ser >= 3.10

# Instalar dependencias
pip install numpy matplotlib

# Verificar que funciona
python3 paloma_sim.py --list
```

---

## 6. Uso desde línea de comandos

### Referencia completa de parámetros

```
python paloma_sim.py [opciones]

Parámetros de algoritmo:
  --algorithm ARCHIVO, -a   Archivo JSON con el algoritmo a ejecutar.
                            Si se omite, ejecuta el polinomio vectorial original.

Parámetros de la malla:
  --rows N, -r              Número de filas (etapas del pipeline).
  --cols N, -c              Número de columnas (paralelismo espacial).
  --N N, -n                 Longitud de la secuencia temporal.
  --M N, -m                 Número de canales espaciales activos.

Los parámetros de la malla tienen prioridad sobre los defaults del JSON.
Si no se especifican, se usan los defaults definidos en el archivo JSON.

Opciones de ejecución:
  --seed N, -s              Semilla del generador de números aleatorios (default: 42).
  --output ARCHIVO, -o      Ruta del dashboard PNG (default: paloma_simulation.png).
  --no-dashboard            Omitir la generación del dashboard visual.
  --quiet, -q               Reducir la salida por consola.
  --list, -l                Listar los algoritmos disponibles en ./algorithms/.
  --help, -h                Mostrar esta ayuda.
```

### Ejemplos representativos

```bash
# Benchmark original (polinomio vectorial de la memoria técnica)
python paloma_sim.py

# Polinomio con malla más grande y más muestras
python paloma_sim.py --rows 8 --cols 32 --N 1000 --M 32

# Ejecutar un algoritmo externo
python paloma_sim.py --algorithm algorithms/rms.json

# Ejecutar con parámetros distintos a los defaults del JSON
python paloma_sim.py --algorithm algorithms/dft_spectrum.json --N 1024 --M 8

# Sin dashboard (útil para benchmarks o scripting)
python paloma_sim.py --algorithm algorithms/signal_power_db.json --no-dashboard --quiet

# Ver todos los algoritmos disponibles
python paloma_sim.py --list

# Guardar el dashboard con nombre específico
python paloma_sim.py -a algorithms/fourier_synthesis.json -o fourier_dashboard.png
```

---

## 7. El formato JSON de algoritmos

Cada algoritmo se define en un archivo JSON con seis secciones. El diseño refleja directamente el modelo de programación de PALOMA: primero se describe qué datos entran, luego qué hace cada fila del pipeline, y finalmente cómo se recoge la salida.

### Sección `metadata`

```json
"metadata": {
  "name": "Nombre descriptivo del algoritmo",
  "description": "Descripción extendida de qué calcula y por qué.",
  "reference_formula": "E_j = Σ |A_j · sin(...)| — expresión matemática compacta",
  "complexity": "N etapas: INSTRUCCIÓN → INSTRUCCIÓN → ...",
  "opcodes_used": ["MULA", "SIN", "ABS"]
}
```

### Sección `defaults`

Valores por defecto de los parámetros de la malla. Los flags CLI tienen prioridad.

```json
"defaults": { "rows": 8, "cols": 16, "N": 512, "M": 16 }
```

### Sección `sequence`

Define la señal temporal de entrada `x[0..N-1]`. El campo `type` puede ser `random_uniform`, `random_normal`, `linspace`, `sine`, `ones`, `zeros`, o `const`.

```json
"sequence": { "type": "sine", "frequency": 4.0, "amplitude": 1.0, "phase": 0.0 }
```

```json
"sequence": { "type": "random_normal", "mean": 0.0, "std": 1.5, "seed": 99 }
```

### Sección `spatial`

Vectores de constantes por canal `[j=0..M-1]`. Cada elemento define un array con los mismos tipos que `sequence`, más un campo obligatorio `name` por el que se referencia desde el pipeline.

```json
"spatial": [
  { "name": "W",  "type": "random_uniform", "range": [0.5, 2.0], "seed": 77 },
  { "name": "MU", "type": "const", "value": 2.5 }
]
```

### Sección `derived_spatial` (opcional)

Arrays espaciales calculados mediante expresiones numpy a partir de otros arrays y de los parámetros `N` y `M`. Esto es esencial para algoritmos cuyas constantes dependen del tamaño de la ventana, como el paso de fase de la DFT (`Δθ = 2π·FC/N`).

```json
"derived_spatial": [
  { "name": "FREQ_STEP", "numpy_expr": "FC * 2 * math.pi / N" },
  { "name": "PHASE_INIT", "numpy_expr": "-FC * 2 * math.pi / N" }
]
```

En las expresiones están disponibles `np`, `math`, `N`, `M`, y todos los arrays definidos en `spatial` por su nombre.

### Sección `pipeline`

El corazón del formato. Cada elemento define una fila de la malla: la instrucción que ejecuta ese PE y de dónde provienen sus operandos. El orden de los elementos marca el orden de dependencia en el grafo de flujo de datos.

```json
"pipeline": [
  {
    "stage": 0,
    "instruction": "MUL",
    "description": "Texto libre — ignorado por el motor",
    "rega_source": "sequence",
    "regb_source": "sequence",
    "init_val": 0.0,
    "operand": 0.0
  },
  {
    "stage": 1,
    "instruction": "MULA",
    "rega_source": "const:1.0",
    "regb_source": "stage:0",
    "init_val": 0.0,
    "preload_result": "PHASE_INIT"
  }
]
```

Los valores válidos para `rega_source` y `regb_source` son los siguientes. Entender esta tabla es entender el modelo de programación de PALOMA:

| Valor | Significado |
|---|---|
| `"sequence"` | Valor actual `x[i]` de la secuencia temporal |
| `"spatial:NAME"` | Constante del vector espacial `NAME[j]` para el canal actual |
| `"const:VALUE"` | Literal numérico, e.g. `"const:1.0"` o `"const:-0.5"` |
| `"stage:N"` | Resultado de la etapa N en este mismo ciclo de máquina |
| `"accumulator"` | Valor actual de `RESULT` del PE (para inspección sin modificarlo) |
| `null` | Cero |

El campo `preload_result` acepta el nombre de un array espacial (incluyendo `derived_spatial`) y carga `INITVAL` del PE con el valor `array[j]` específico de ese canal. Es imprescindible para acumuladores de fase donde cada canal arranca en un ángulo inicial distinto.

### Sección `output`

```json
"output": {
  "stage": 5,
  "accumulate_in_pe": true,
  "postprocess": "sqrt_divide_N"
}
```

`accumulate_in_pe: true` indica que el resultado final está en `PE(stage, j).RESULT` al terminar. Es el caso de las etapas MULA que acumulan ciclo a ciclo. `accumulate_in_pe: false` hace que el simulador sume externamente los valores de salida de la etapa en cada ciclo (útil para etapas de paso que no acumulan en el PE).

Los postprocesados disponibles son:

| Valor | Operación |
|---|---|
| `null` o `"identity"` | Sin transformación |
| `"abs"` | `\|acc\|` |
| `"divide_N"` | `acc / N` |
| `"abs_divide_N"` | `\|acc\| / N` |
| `"sqrt"` | `sqrt(\|acc\|)` |
| `"sqrt_divide_N"` | `sqrt(\|acc\| / N)` |
| `"negate"` | `-acc` |
| `"log10_scale10"` | `10 · log10(\|acc\|)` (el acumulador ya es una media) |
| `"divide_N_then_dB"` | `10 · log10(\|acc\| / N)` (divide antes del logaritmo) |

### Sección `reference`

Expresión numpy que calcula el resultado correcto de forma secuencial. El motor la evalúa y compara con la salida del simulador para calcular el error relativo medio. Están disponibles `np`, `math`, `x`, `N`, `M` y todos los arrays espaciales por nombre.

```json
"reference": {
  "numpy_expr": "np.array([np.sqrt(np.mean((W[j]*x)**2)) for j in range(M)])"
}
```

---

## 8. Algoritmos incluidos

Los ocho algoritmos de ejemplo cubren una gama representativa de cómputo numérico intensivo y demuestran de forma práctica distintas instrucciones del repertorio.

**`rms.json` — RMS con Ponderación por Canal** implementa `rms_j = sqrt((1/N)·Σ(W_j·x_i)²)`. Pipeline de 3 etapas (MUL → SQR → MULA) con una señal senoidal de entrada y ponderaciones aleatorias por canal. Es el algoritmo más sencillo del conjunto y sirve como punto de partida para entender el formato.

**`signal_power_db.json` — Potencia en dBW por Canal** calcula `P_dBW_j = 10·log10((1/N)·Σ(G_j·x_i)²)`. Demuestra el pipeline MUL→SQR→MULA seguido del postprocesado `divide_N_then_dB`, que aplica la división por N correctamente antes del logaritmo (error común al invertir el orden).

**`weighted_variance.json` — Varianza Ponderada Multi-Canal** calcula `var_j = (1/N)·Σ W_j·(x_i−μ_j)²` en un pipeline de 6 etapas (MUL→MULA→SUB→SQR→MUL→MULA). Ilustra el patrón de dos pasadas sobre el mismo dataset mapeadas en la mitad superior e inferior del pipeline, con la media μ_j cargada como constante espacial.

**`dft_spectrum.json` — Espectro DFT, Proyección Coseno** implementa `S_j = |(1/N)·Σ x_i·cos(2π·FC_j·i/N)|`. Es el algoritmo más técnico del conjunto porque requiere el patrón de **PE generador de fase**: la etapa 0 es un acumulador MULA cuyo RESULT crece linealmente como `i·Δθ_j`, simulando un oscilador de fase. El truco del `PHASE_INIT = -FREQ_STEP_j` compensa el off-by-one del acumulador para que en `i=0` el coseno valga exactamente 1. Usa `derived_spatial` para calcular el paso de fase adaptado al valor de N.

**`fourier_synthesis.json` — Síntesis de Fourier Multi-Canal** calcula `E_j = Σ|A_j·sin(2π·f_j·i/N+φ_j)|`. Mismo patrón de acumulador de fase que `dft_spectrum`, ahora con SIN en lugar de COS y fases iniciales distintas por canal (cargadas via `preload_result`). Acumula la norma L1 de la forma de onda sinusoidal en lugar de su integral (que se anularía por simetría).

**`hyperbolic_norm.json` — Norma Hiperbólica Multi-Canal** calcula `d_j = (1/N)·Σ SCALE_j·sqrt((x_i−R_j)²+ε)`. Demuestra SQRT del grupo 7 (latencia 8 ciclos) en el interior del pipeline, no solo al final. La constante de regularización `ε=1e-6` evita `sqrt(0)` exacto.

**`log_energy.json` — Energía Logarítmica por Canal** calcula `E_j = Σ ln(1+G_j·|x_i|)`. Demuestra LN del grupo 7 (latencia 16 ciclos) con el pipeline ABS→MUL→INC→LN→MULA. La función logarítmica es robusta a picos esporádicos y útil en detección de actividad vocal y compresión perceptual.

**`mahalanobis.json` — Distancia de Mahalanobis Estandarizada** calcula `D_j = (1/N)·Σ min(sqrt((x_i−μ_j)²/σ_j²), CAP)`. Usa 6 etapas (SUB→SQR→DIV→SQRT→MIN→MULA). La instrucción MIN actúa como acotador de outliers sin necesidad de bifurcaciones — patrón canónico del modelo dataflow de PALOMA donde la lógica condicional se reemplaza por saturación diferenciable.

---

## 9. Usar la API Python directamente

Se puede importar el módulo y programar algoritmos arbitrarios desde Python sin pasar por el formato JSON:

```python
from paloma_sim import MMI, PE, Opcode

# Crear una malla 4×4
mmi = MMI(rows=4, cols=4)
mmi.reset()

# Programar todos los PEs con ADD
for r in range(mmi.rows):
    for c in range(mmi.cols):
        p = mmi.pe(r, c)
        p.INSTRUCT = Opcode.ADD
        p.REGA     = float(r * 10 + c)
        p.REGB     = 1.0
        p.restart()

# Ejecutar 10 ciclos de máquina manualmente
for cycle in range(10):
    max_lat = 1
    for r in range(mmi.rows):
        for c in range(mmi.cols):
            p = mmi.pe(r, c)
            p.execute()
            p.tick_counter()
    mmi.tick(max_lat)

# Leer resultados y métricas
for r in range(mmi.rows):
    for c in range(mmi.cols):
        print(f"PE({r},{c}).RESULT = {mmi.pe(r,c).RESULT:.1f}")

print(f"Potencia pico: {mmi.peak_gflops:.4f} GFLOPS")
print(f"Tiempo estimado: {mmi.elapsed_ns:.1f} ns")
```

Para cargar y ejecutar un algoritmo JSON directamente desde Python:

```python
import json
from paloma_sim import AlgorithmRunner

with open("algorithms/rms.json") as f:
    algo_def = json.load(f)

# rows/cols/N/M: None → usa los defaults del JSON
runner = AlgorithmRunner(algo_def, rows=None, cols=None, N=1024, M=8)
result = runner.run(seed=42, verbose=True)

# El dict devuelto contiene todos los arrays y métricas
print(f"Error vs referencia: {result['error_rel_pct']:.4f} %")
print(f"Ciclos de máquina: {result['machine_cycles']}")
y = result["y_paloma"]   # array numpy con los resultados por canal
```

---

## 10. Personalización y extensión

### Añadir nuevas instrucciones

Aunque el repertorio ya cubre los 64 slots del OPCODE, si se quisiera redefinir alguno se modifican tres sitios: el enum `Opcode`, el diccionario `LATENCY`, y el `match` dentro de `PE.execute()`. Las tres modificaciones son simétricas e inmediatamente verificables contra la referencia numpy de cualquier algoritmo existente.

### Añadir nuevos tipos de secuencia

El generador de datos `generate_array()` soporta los tipos más comunes. Para añadir uno nuevo (por ejemplo señales de chirp o ruido rosa) basta con añadir una rama `elif t == "chirp":` al final de la función y definir los parámetros en la sección `sequence` del JSON.

### Añadir nuevos postprocesados

El método `_postprocess()` de `AlgorithmRunner` es una tabla de casos con una línea por operación. Para añadir uno nuevo se añade una línea `elif pp == "mi_op": return ...` antes del `else: raise ValueError`.

### Visualizar la malla en tiempo real

Para depurar algoritmos pequeños, `matplotlib.animation` permite ver el estado de la malla ciclo a ciclo:

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from paloma_sim import MMI, Opcode

mmi = MMI(8, 8)
# ... programar la malla ...

fig, ax = plt.subplots()

def update(frame):
    # Ejecutar un ciclo manualmente
    for r in range(mmi.rows):
        for c in range(mmi.cols):
            mmi.pe(r, c).execute()
            mmi.pe(r, c).tick_counter()
    mmi.tick(1)
    heat = [[mmi.pe(r,c).RESULT for c in range(8)] for r in range(8)]
    ax.clear()
    ax.imshow(heat, cmap='coolwarm', vmin=-2, vmax=2)
    ax.set_title(f"Ciclo {mmi.machine_cycles}")

ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
plt.show()
```

---

## 11. Repertorio completo de 64 instrucciones

El OPCODE tiene 6 bits, lo que permite exactamente 64 operaciones. Todas están implementadas con sus latencias reales extraídas de la especificación del PE. Las operaciones del Grupo 7 son extensiones matemáticas cuyos slots en el OPCODE estaban reservados en el documento original; las latencias son estimaciones razonables para una FPU hardware de la época.

**Grupo 0 — Control (1 instrucción)**

`NOP` (1 ciclo): sin operación, el PE devuelve RESULT sin modificarlo.

**Grupo 1 — Aritmética (12 instrucciones)**

`ADD` (1), `SUB` (1), `MUL` (4), `DIV` (32), `MOD` (8), `NEG` (1), `MULA` (5), `ABS` (1), `MAX` (2), `MIN` (2), `INC` (1), `DCR` (1).

La instrucción `MULA` (*multiply-accumulate*) es la más importante del repertorio: implementa `R ← R + op1 × op2` en un solo ciclo de máquina, que es el kernel de álgebra lineal más frecuente (productos de matrices, filtros FIR, correlaciones). Su latencia de 5 ciclos es solo 1 más que MUL porque la suma final se solapa con la multiplicación en pipeline interno.

**Grupo 2 — Lógica (4 instrucciones)**

`AND`, `OR`, `XOR`, `NOT` (todas 1 ciclo). Operan sobre la representación entera de 32 bits del operando.

**Grupo 3 — Comparación (7 instrucciones)**

`TST`, `GRT`, `LWR`, `EQU`, `NEQ`, `GEQ`, `LEQ` (todas 1 ciclo). Actualizan el flag `zero` de FLAGS pero no modifican RESULT. Son útiles para programar PEs que generan señales de control o máscaras booleanas que fluyen hacia otros PEs.

**Grupo 4 — Desplazamiento y rotación (6 instrucciones)**

`ROL`, `ROR` (2 ciclos): rotación de 1 bit a la izquierda/derecha. `SHL`, `SHR` (2 ciclos): desplazamiento de 1 bit. `ROT`, `SHF` (3 ciclos): rotación/desplazamiento de N bits (N en REGB).

**Grupo 5 — Conversión y retardos de pipeline (7 instrucciones)**

`CPY` (1): copia REGA a RESULT. `DEL2` (1): línea de retardo de 1 ciclo; RESULT recibe el valor del ciclo anterior. `DEL3` (1): línea de retardo de 2 ciclos. Estas dos instrucciones permiten alinear temporalmente señales que viajan por ramas del pipeline con distinto número de etapas. `BYTE` (1), `WORD` (1): extensión de signo desde 8/16 bits. `INT` (4), `FLT` (4): conversión entre representación entera y flotante.

**Grupo 6 — Entrada/Salida y RAM (4 instrucciones)**

`LD`, `ST` (33 ciclos cada una): lectura y escritura en la RAM compartida de la tarjeta. La latencia de 33 ciclos corresponde al tiempo de transmisión serie del bus de datos (32 bits + paridad). `IN`, `OUT` (33 ciclos): acceso a puertos de entrada/salida.

**Grupo 7 — Extensiones matemáticas (23 instrucciones)**

`SQRT` (8), `SQR` (4), `EXP` (16), `LN` (16), `LOG2` (16), `LOG10` (16), `SIN` (20), `COS` (20), `TAN` (24), `ASIN` (24), `ACOS` (24), `ATAN` (16), `ATAN2` (20), `POW` (24), `HYPOT` (12), `CEIL` (2), `FLOOR` (2), `ROUND` (2), `TRUNC` (2), `CLAMP` (2), `LERP` (5), `FMOD` (8), `SGN` (1).

`HYPOT` calcula `sqrt(a²+b²)` en 12 ciclos sin desbordamiento intermedio —imprescindible para cálculos de módulo complejo en el interior de pipelines DFT—. `LERP` interpola linealmente entre dos valores usando OPERAND como factor, lo que permite programar interpoladores y mezcladores sin etapas adicionales. `CLAMP` satura RESULT entre REGA y REGB, que junto con `MIN`/`MAX` del Grupo 1 permite implementar lógica condicional de forma diferenciable en el modelo dataflow.

---

## 12. Constantes arquitectónicas y sus fuentes

Todas las constantes del simulador provienen directamente de los documentos técnicos de Prototec S.L.:

| Constante | Valor | Fuente |
|---|---|---|
| `CLOCK_FREQ_MHZ` | 66 MHz | Memoria técnica, pág. 21: "frecuencia de reloj de 66MHz, el doble de la frecuencia del bus PCI" |
| `TRANSMIT_CYCLES` | 33 | Memoria técnica, pág. 19: "los registros tienen 32 bits y se envía un bit de paridad, se requieren siempre 33 ciclos" |
| `MACHINE_CYCLE` | 64 | Especificación PE, pág. 9: "el ciclo de máquina típico dura entre 50 y 64 ciclos de reloj" |
| `MFLOPS_PER_PE` | 2.0 | Memoria técnica, pág. 21: "cada procesador elemental puede ejecutar unos dos megaflops en aplicaciones reales" |
| `CPU_REF_GFLOPS` | 0.4 | Referencia histórica: Pentium II @400 MHz (~1998) |

---

## 13. Limitaciones del modelo

Este simulador es **behavioral**, no **RTL** (*Register Transfer Level*). Las limitaciones más relevantes son las siguientes.

La simulación de la malla es **secuencial** en Python: los PEs no se ejecutan en paralelo real, sino iterando sobre la malla en orden fila-columna. Las métricas de tiempo y GFLOPS son estimaciones correctas del hardware real, pero el tiempo de simulación en Python crece linealmente con el número de PEs y de muestras.

Los **accesos a RAM** están simplificados: `LD` y `ST` acumulan latencia correctamente pero no realizan transferencias reales al diccionario `mmi.ram`. El arbitraje por daisy chain del bus de datos (`BUSBUSY`/`BUSHOLD`) no está implementado.

Las **células de interconexión programables** (`CELLPRG`, `IOPRG`) no están modeladas. La transmisión de datos entre vecinos usa el modelo behavioral de fuentes de operandos declaradas en el JSON, que captura el flujo del dataflow pero no la programabilidad completa de las puertas MUX 4:1 de la red hardware real.

El **tiempo estimado** (`elapsed_ns`) usa el peor caso de latencia por fase y no refleja la optimización que el programador puede hacer reprogramando `CLKTX` según el máximo retardo de propagación del algoritmo concreto.

---

## 14. Referencias

Los documentos técnicos en los que se basa este simulador son los siguientes, todos conservados en el archivo de la comunidad **El Mundo del Spectrum**. 

https://www.elmundodelspectrum.com/descubrimos-proyecto-paloma-el-visionario-trabajo-al-que-paco-menendez-dedico-sus-ultimos-anos-de-vida/

**Proyecto Paloma — Descripción de la arquitectura** (presentación). Prototec S.L., ~1998. 9 transparencias. Cubre la topología de la malla, el chip MMI, la tarjeta MMI y la PLM.

**Proyecto Paloma — Memoria técnica y Plan de desarrollo y financiero**. Prototec S.L., ~1998. 27 páginas. Cubre la génesis del proyecto, la descripción técnica completa (modelo de flujo de datos, topología, celda elemental MMI, ciclo de máquina, intérprete de comandos, chip MMI, tarjeta MMI, PLM), el desarrollo del proyecto y el plan de negocio.

**Proyecto Paloma — Especificación del procesador elemental**. Prototec S.L., ~1998. 19 páginas. Cubre la especificación funcional completa (registros internos, repertorio de instrucciones, ciclo de máquina) y la especificación hardware del PE (interfaz, células de interconexión, unidad funcional, FPALU y unidad de control con estimaciones de complejidad en puertas lógicas).

---

*Simulador desarrollado con fines históricos y educativos. El Proyecto PALOMA fue una propuesta técnicamente sólida que no llegó a fabricarse por razones de financiación, no de viabilidad arquitectónica. Sus ideas fundamentales sobre escalabilidad lineal mediante comunicación local son hoy la base de algunas de las arquitecturas de aceleración de IA más potentes del mundo.*
