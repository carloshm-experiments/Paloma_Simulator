"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROYECTO PALOMA — Parallel Logic Machine  ·  Simulador Python v2.0        ║
║  Prototec S.L. (~1998)                                                      ║
║                                                                              ║
║  Novedades v2:                                                               ║
║   · Repertorio completo de 64 instrucciones (6 bits OPCODE)                ║
║   · Parámetros de ejecución vía CLI (argparse)                              ║
║   · Carga de algoritmos desde archivos JSON externos                        ║
║   · Modelo behavioral: cada etapa declara explícitamente sus fuentes       ║
║     de operandos → simula fielmente el dataflow graph de PALOMA             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Uso:
  python paloma_sim.py                              # benchmark original
  python paloma_sim.py --rows 16 --cols 16 --N 500
  python paloma_sim.py --algorithm algorithms/rms.json
  python paloma_sim.py --list
  python paloma_sim.py --help
"""

import argparse, json, math, time, sys, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES ARQUITECTÓNICAS
# Fuente: Memoria técnica Prototec S.L. + especificación del PE
# ──────────────────────────────────────────────────────────────────────────────

CLOCK_FREQ_MHZ  = 66      # 66 MHz — el doble de la frecuencia del bus PCI
TRANSMIT_CYCLES = 33      # 32 bits de datos + 1 bit de paridad en serie
MACHINE_CYCLE   = 64      # ciclos de reloj típicos por ciclo de máquina
MFLOPS_PER_PE   = 2.0     # rendimiento real por PE (memoria técnica, pág. 21)
CPU_REF_GFLOPS  = 0.4     # Pentium II ~400 MFLOPS — referencia año 1998

# Directorio de algoritmos: junto al propio script
ALG_DIR = Path(__file__).parent / "algorithms"


# ──────────────────────────────────────────────────────────────────────────────
# REPERTORIO COMPLETO — 64 INSTRUCCIONES (OPCODE de 6 bits)
#
# Grupos según la especificación del procesador elemental:
#   Grupo 0  (opcode  0):       Control                 →  1 instrucción
#   Grupo 1  (opcodes 1-12):    Aritmética              → 12 instrucciones
#   Grupo 2  (opcodes 13-16):   Lógica                  →  4 instrucciones
#   Grupo 3  (opcodes 17-23):   Comparación             →  7 instrucciones
#   Grupo 4  (opcodes 24-29):   Desplazamiento/rotación →  6 instrucciones
#   Grupo 5  (opcodes 30-36):   Conversión y retardos   →  7 instrucciones
#   Grupo 6  (opcodes 37-40):   Entrada/Salida/RAM      →  4 instrucciones
#   Grupo 7  (opcodes 41-63):   Extensiones matemáticas → 23 instrucciones
#                                                          ──────────────────
#                                                TOTAL  → 64 instrucciones
# ──────────────────────────────────────────────────────────────────────────────

class Opcode(Enum):
    # ── Grupo 0: Control ──────────────────────────────────────────────────────
    NOP   = "NOP"    # Sin operación

    # ── Grupo 1: Aritmética (especificación pág. 6-7) ─────────────────────────
    ADD   = "ADD"    # R ← op1 + op2
    SUB   = "SUB"    # R ← op1 − op2
    MUL   = "MUL"    # R ← op1 × op2
    DIV   = "DIV"    # R ← op1 / op2
    MOD   = "MOD"    # R ← op1 mod op2 (fija) / parte decimal (flotante)
    NEG   = "NEG"    # R ← −op1
    MULA  = "MULA"   # R ← R + op1 × op2  ← multiply-accumulate clave de PALOMA
    ABS   = "ABS"    # R ← |op1|
    MAX   = "MAX"    # R ← max(op1, op2)
    MIN   = "MIN"    # R ← min(op1, op2)
    INC   = "INC"    # R ← op1 + 1
    DCR   = "DCR"    # R ← op1 − 1

    # ── Grupo 2: Lógica ───────────────────────────────────────────────────────
    AND   = "AND"    # R ← op1 AND op2
    OR    = "OR"     # R ← op1 OR  op2
    XOR   = "XOR"    # R ← op1 XOR op2
    NOT   = "NOT"    # R ← NOT op1

    # ── Grupo 3: Comparación (actualizan flag Z; RESULT queda inalterado) ─────
    TST   = "TST"    # Z ← (op1 == 0)
    GRT   = "GRT"    # Z ← (op1 > op2)
    LWR   = "LWR"    # Z ← (op1 < op2)
    EQU   = "EQU"    # Z ← (op1 == op2)
    NEQ   = "NEQ"    # Z ← (op1 != op2)
    GEQ   = "GEQ"    # Z ← (op1 >= op2)
    LEQ   = "LEQ"    # Z ← (op1 <= op2)

    # ── Grupo 4: Desplazamiento y rotación ────────────────────────────────────
    ROL   = "ROL"    # R ← rotate_left_1(op1)
    ROR   = "ROR"    # R ← rotate_right_1(op1)
    SHL   = "SHL"    # R ← shift_left_1(op1)
    SHR   = "SHR"    # R ← shift_right_1(op1)
    ROT   = "ROT"    # R ← rotate_N_bits(op1, op2)
    SHF   = "SHF"    # R ← shift_N_bits(op1, op2)

    # ── Grupo 5: Conversión y retardos de pipeline ────────────────────────────
    CPY   = "CPY"    # R ← op1
    DEL2  = "DEL2"   # R ← delay_buf; delay_buf ← op1  (retardo 1 ciclo)
    DEL3  = "DEL3"   # Retardo 2 ciclos (buffer doble)
    BYTE  = "BYTE"   # R ← sign_extend_byte(op1)
    WORD  = "WORD"   # R ← sign_extend_word(op1)
    INT   = "INT"    # R ← int(op1)    conversión flotante → fija
    FLT   = "FLT"    # R ← float(op1) conversión fija → flotante

    # ── Grupo 6: Entrada/Salida y memoria RAM ─────────────────────────────────
    LD    = "LD"     # R ← RAM[ADDR]; ADDR += OPERAND
    ST    = "ST"     # RAM[ADDR] ← op1; ADDR += OPERAND
    IN    = "IN"     # R ← Puerto[ADDR]
    OUT   = "OUT"    # Puerto[ADDR] ← op1

    # ── Grupo 7: Funciones matemáticas extendidas ─────────────────────────────
    SQRT  = "SQRT"   # R ← √|op1|
    SQR   = "SQR"    # R ← op1²
    EXP   = "EXP"    # R ← eˣ
    LN    = "LN"     # R ← ln|op1|
    LOG2  = "LOG2"   # R ← log₂|op1|
    LOG10 = "LOG10"  # R ← log₁₀|op1|
    SIN   = "SIN"    # R ← sin(op1)  [radianes]
    COS   = "COS"    # R ← cos(op1)
    TAN   = "TAN"    # R ← tan(op1)
    ASIN  = "ASIN"   # R ← arcsin(clamp(op1, -1, 1))
    ACOS  = "ACOS"   # R ← arccos(clamp(op1, -1, 1))
    ATAN  = "ATAN"   # R ← arctan(op1)
    ATAN2 = "ATAN2"  # R ← arctan(op1 / op2)
    POW   = "POW"    # R ← |op1|^op2
    HYPOT = "HYPOT"  # R ← √(op1² + op2²)
    CEIL  = "CEIL"   # R ← ⌈op1⌉
    FLOOR = "FLOOR"  # R ← ⌊op1⌋
    ROUND = "ROUND"  # R ← round(op1)
    TRUNC = "TRUNC"  # R ← trunc(op1)
    CLAMP = "CLAMP"  # R ← clamp(RESULT, op1, op2)
    LERP  = "LERP"   # R ← op1 + (op2 − op1) × OPERAND
    FMOD  = "FMOD"   # R ← fmod(op1, op2)
    SGN   = "SGN"    # R ← sign(op1): −1.0, 0.0, o 1.0


# Latencia de cada instrucción en ciclos de reloj interno.
# Los valores de la especificación son: suma/resta=1, mul=4, div=32.
# Las funciones trascendentales del grupo 7 son estimaciones para FPU hardware.
LATENCY: Dict[Opcode, int] = {
    Opcode.NOP: 1,
    Opcode.ADD: 1,  Opcode.SUB: 1,  Opcode.MUL: 4,   Opcode.DIV: 32,
    Opcode.MOD: 8,  Opcode.NEG: 1,  Opcode.MULA: 5,  Opcode.ABS: 1,
    Opcode.MAX: 2,  Opcode.MIN: 2,  Opcode.INC: 1,   Opcode.DCR: 1,
    Opcode.AND: 1,  Opcode.OR:  1,  Opcode.XOR: 1,   Opcode.NOT: 1,
    Opcode.TST: 1,  Opcode.GRT: 1,  Opcode.LWR: 1,   Opcode.EQU: 1,
    Opcode.NEQ: 1,  Opcode.GEQ: 1,  Opcode.LEQ: 1,
    Opcode.ROL: 2,  Opcode.ROR: 2,  Opcode.SHL: 2,   Opcode.SHR: 2,
    Opcode.ROT: 3,  Opcode.SHF: 3,
    Opcode.CPY: 1,  Opcode.DEL2: 1, Opcode.DEL3: 1,
    Opcode.BYTE: 1, Opcode.WORD: 1, Opcode.INT: 4,   Opcode.FLT: 4,
    Opcode.LD: 33,  Opcode.ST: 33,  Opcode.IN: 33,   Opcode.OUT: 33,
    Opcode.SQRT: 8,   Opcode.SQR: 4,   Opcode.EXP: 16,
    Opcode.LN: 16,    Opcode.LOG2: 16,  Opcode.LOG10: 16,
    Opcode.SIN: 20,   Opcode.COS: 20,   Opcode.TAN: 24,
    Opcode.ASIN: 24,  Opcode.ACOS: 24,  Opcode.ATAN: 16,  Opcode.ATAN2: 20,
    Opcode.POW: 24,   Opcode.HYPOT: 12,
    Opcode.CEIL: 2,   Opcode.FLOOR: 2,  Opcode.ROUND: 2,  Opcode.TRUNC: 2,
    Opcode.CLAMP: 2,  Opcode.LERP: 5,   Opcode.FMOD: 8,   Opcode.SGN: 1,
}


# ──────────────────────────────────────────────────────────────────────────────
# ELEMENTO DE PROCESO (PE)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PE:
    """
    Procesador Elemental de la arquitectura PALOMA.

    Los registros delay_buf2/3 implementan las líneas de retardo de las
    instrucciones DEL2 y DEL3: son internos al PE y no se exponen al bus.
    """
    row: int
    col: int

    REGA:     float = 0.0
    REGB:     float = 0.0
    REGC:     float = 0.0
    REGD:     float = 0.0
    RESULT:   float = 0.0
    OPERAND:  float = 0.0

    # Buffers de retardo para DEL2/DEL3
    delay_buf2:  float = 0.0
    delay_buf3a: float = 0.0
    delay_buf3b: float = 0.0

    # Registros de programa (fijos durante la ejecución)
    INSTRUCT: Optional[Opcode] = None
    INITVAL:  float = 0.0
    INITCNT:  int   = 0
    MAXCNT:   int   = 1
    MEMBASE:  int   = 0

    # Estado de control
    ENACNT: int = 0
    ADDR:   int = 0
    FLAGS:  Dict = field(default_factory=lambda: {"zero": False, "overflow": False, "carry": False})

    # Estadísticas
    exec_count:  int = 0
    cycles_used: int = 0

    def restart(self) -> None:
        """Carga INITVAL en RESULT y reinicia contadores (señal RESTART del bus)."""
        self.RESULT     = self.INITVAL
        self.ENACNT     = self.INITCNT
        self.ADDR       = self.MEMBASE
        self.delay_buf2 = 0.0
        self.delay_buf3a = 0.0
        self.delay_buf3b = 0.0
        self.FLAGS = {"zero": False, "overflow": False, "carry": False}

    def is_enabled(self) -> bool:
        return self.ENACNT == 0

    def tick_counter(self) -> None:
        """Avanza el contador de habilitación al final de cada ciclo de máquina."""
        self.ENACNT = (self.ENACNT + 1) % max(1, self.MAXCNT)

    def execute(self) -> float:
        """
        Fase 1 del ciclo de máquina: ejecución de la instrucción programada.
        Si ENACNT != 0 el PE está inhibido y devuelve RESULT sin modificarlo.
        """
        if not self.is_enabled() or self.INSTRUCT is None:
            return self.RESULT

        op = self.INSTRUCT
        a  = self.REGA
        b  = self.REGB
        ia = int(a) & 0xFFFFFFFF
        ib = int(b) & 0xFFFFFFFF

        def safe_div(x, y):
            return (x / y) if y != 0.0 else (math.copysign(float("inf"), x) if x else 0.0)

        def safe_log(x, base=math.e):
            if x == 0.0: return -math.inf
            return math.log(abs(x)) / math.log(base)

        def clamp_f(v, lo, hi):
            return max(lo, min(hi, v))

        match op:
            # ── Grupo 0
            case Opcode.NOP:   r = self.RESULT
            # ── Grupo 1: Aritmética
            case Opcode.ADD:   r = a + b
            case Opcode.SUB:   r = a - b
            case Opcode.MUL:   r = a * b
            case Opcode.DIV:   r = safe_div(a, b)
            case Opcode.MOD:   r = math.fmod(a, b) if b != 0.0 else 0.0
            case Opcode.NEG:   r = -a
            case Opcode.MULA:  r = self.RESULT + a * b   # acumula en RESULT previo
            case Opcode.ABS:   r = abs(a)
            case Opcode.MAX:   r = max(a, b)
            case Opcode.MIN:   r = min(a, b)
            case Opcode.INC:   r = a + 1.0
            case Opcode.DCR:   r = a - 1.0
            # ── Grupo 2: Lógica (sobre representación entera de 32 bits)
            case Opcode.AND:   r = float(ia & ib)
            case Opcode.OR:    r = float(ia | ib)
            case Opcode.XOR:   r = float(ia ^ ib)
            case Opcode.NOT:   r = float((~ia) & 0xFFFFFFFF)
            # ── Grupo 3: Comparación (no modifican RESULT)
            case Opcode.TST:
                self.FLAGS["zero"] = (a == 0.0); return self.RESULT
            case Opcode.GRT:
                self.FLAGS["zero"] = (a > b); return self.RESULT
            case Opcode.LWR:
                self.FLAGS["zero"] = (a < b); return self.RESULT
            case Opcode.EQU:
                self.FLAGS["zero"] = (a == b); return self.RESULT
            case Opcode.NEQ:
                self.FLAGS["zero"] = (a != b); return self.RESULT
            case Opcode.GEQ:
                self.FLAGS["zero"] = (a >= b); return self.RESULT
            case Opcode.LEQ:
                self.FLAGS["zero"] = (a <= b); return self.RESULT
            # ── Grupo 4: Desplazamiento y rotación
            case Opcode.ROL:
                r = float(((ia << 1) | (ia >> 31)) & 0xFFFFFFFF)
            case Opcode.ROR:
                r = float(((ia >> 1) | ((ia & 1) << 31)) & 0xFFFFFFFF)
            case Opcode.SHL:   r = float((ia << 1) & 0xFFFFFFFF)
            case Opcode.SHR:   r = float((ia >> 1) & 0xFFFFFFFF)
            case Opcode.ROT:
                n = int(b) & 31
                r = float(((ia << n) | (ia >> (32 - n))) & 0xFFFFFFFF) if n else float(ia)
            case Opcode.SHF:
                n = int(b) & 63
                r = float((ia << n) & 0xFFFFFFFF) if n < 32 else 0.0
            # ── Grupo 5: Conversión y retardos
            case Opcode.CPY:   r = a
            case Opcode.DEL2:
                r = self.delay_buf2; self.delay_buf2 = a
            case Opcode.DEL3:
                r = self.delay_buf3b
                self.delay_buf3b = self.delay_buf3a
                self.delay_buf3a = a
            case Opcode.BYTE:
                sign = (ia >> 7) & 1
                r = float((ia & 0xFF) - 256 if sign else ia & 0xFF)
            case Opcode.WORD:
                sign = (ia >> 15) & 1
                r = float((ia & 0xFFFF) - 65536 if sign else ia & 0xFFFF)
            case Opcode.INT:   r = float(int(a))
            case Opcode.FLT:   r = float(int(a))
            # ── Grupo 6: I/O (latencia modelada; sin transferencia real de datos)
            case Opcode.LD:    r = a
            case Opcode.ST:    r = self.RESULT
            case Opcode.IN:    r = a
            case Opcode.OUT:   r = self.RESULT
            # ── Grupo 7: Extensiones matemáticas
            case Opcode.SQRT:  r = math.sqrt(abs(a))
            case Opcode.SQR:   r = a * a
            case Opcode.EXP:
                try:    r = math.exp(clamp_f(a, -700.0, 700.0))
                except: r = math.inf
            case Opcode.LN:    r = safe_log(a)
            case Opcode.LOG2:  r = safe_log(a, 2.0)
            case Opcode.LOG10: r = safe_log(a, 10.0)
            case Opcode.SIN:   r = math.sin(a)
            case Opcode.COS:   r = math.cos(a)
            case Opcode.TAN:
                try:    r = math.tan(a)
                except: r = math.inf
            case Opcode.ASIN:  r = math.asin(clamp_f(a, -1.0, 1.0))
            case Opcode.ACOS:  r = math.acos(clamp_f(a, -1.0, 1.0))
            case Opcode.ATAN:  r = math.atan(a)
            case Opcode.ATAN2: r = math.atan2(a, b)
            case Opcode.POW:
                try:    r = math.pow(abs(a), b)
                except: r = math.inf
            case Opcode.HYPOT: r = math.hypot(a, b)
            case Opcode.CEIL:  r = float(math.ceil(a))
            case Opcode.FLOOR: r = float(math.floor(a))
            case Opcode.ROUND: r = float(round(a))
            case Opcode.TRUNC: r = float(math.trunc(a))
            case Opcode.CLAMP: r = clamp_f(self.RESULT, a, b)
            case Opcode.LERP:  r = a + (b - a) * self.OPERAND
            case Opcode.FMOD:  r = math.fmod(a, b) if b != 0.0 else 0.0
            case Opcode.SGN:   r = (1.0 if a > 0.0 else (-1.0 if a < 0.0 else 0.0))
            case _:            r = self.RESULT

        if not math.isfinite(r):
            r = math.copysign(1e38, r)
        self.RESULT = r
        self.FLAGS["zero"] = (r == 0.0)
        self.exec_count  += 1
        self.cycles_used += LATENCY.get(op, 1)
        return r


# ──────────────────────────────────────────────────────────────────────────────
# MEMORIA MATRICIAL INTELIGENTE (MMI)
# ──────────────────────────────────────────────────────────────────────────────

class MMI:
    """Malla NxM de PEs con su lógica de control y contadores de ciclos."""

    def __init__(self, rows: int, cols: int):
        self.rows    = rows
        self.cols    = cols
        self.num_pes = rows * cols
        self.ram:  Dict[int, float] = {}
        self.grid: List[List[PE]]   = [
            [PE(r, c) for c in range(cols)] for r in range(rows)
        ]
        self.machine_cycles = 0
        self.clock_cycles   = 0

    def pe(self, row: int, col: int) -> PE:
        return self.grid[row % self.rows][col % self.cols]

    def reset(self) -> None:
        for row in self.grid:
            for p in row:
                p.restart()
        self.machine_cycles = 0
        self.clock_cycles   = 0

    def tick(self, max_latency: int) -> None:
        """Contabiliza un ciclo de máquina con solapamiento de fases."""
        self.machine_cycles += 1
        self.clock_cycles   += max(max_latency, TRANSMIT_CYCLES) + TRANSMIT_CYCLES

    @property
    def peak_gflops(self) -> float:
        return (self.num_pes * MFLOPS_PER_PE) / 1000.0

    @property
    def elapsed_ns(self) -> float:
        return self.clock_cycles * (1000.0 / CLOCK_FREQ_MHZ)


# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def generate_array(spec: dict, length: int, seed_override: Optional[int] = None) -> np.ndarray:
    """
    Genera un array de <length> muestras según la especificación del campo 'sequence'
    o 'spatial' de un archivo de algoritmo JSON.
    """
    seed = seed_override if seed_override is not None else spec.get("seed", 42)
    rng  = np.random.default_rng(seed)
    t    = spec.get("type", "random_uniform")
    lo, hi = spec.get("range", [-5.0, 5.0])

    if t == "random_uniform":
        return rng.uniform(lo, hi, length)
    elif t == "random_normal":
        return rng.normal(spec.get("mean", 0.0), spec.get("std", 1.0), length)
    elif t == "linspace":
        return np.linspace(lo, hi, length)
    elif t == "sine":
        freq  = spec.get("frequency", 1.0)
        phase = spec.get("phase", 0.0)
        amp   = spec.get("amplitude", 1.0)
        t_arr = np.linspace(0, 1, length)
        return amp * np.sin(2 * np.pi * freq * t_arr + phase)
    elif t == "ones":
        return np.ones(length)
    elif t == "zeros":
        return np.zeros(length)
    elif t == "const":
        return np.full(length, float(spec.get("value", 1.0)))
    else:
        raise ValueError(f"Tipo de secuencia desconocido: {t!r}")


# ──────────────────────────────────────────────────────────────────────────────
# RESOLUTOR DE FUENTES DE OPERANDOS
# ──────────────────────────────────────────────────────────────────────────────

def resolve_source(
    src: Optional[str],
    xi: float,
    spatial: Dict[str, np.ndarray],
    j: int,
    stage_results: Dict[int, float],
    pe: PE,
) -> float:
    """
    Traduce una cadena descriptiva de fuente de operando al valor float actual.

    Formatos soportados:
      "sequence"      → valor x[i] de la secuencia temporal en curso
      "spatial:NAME"  → constante del vector espacial NAME[j] (por canal)
      "const:VALUE"   → literal numérico
      "stage:N"       → resultado de la etapa N en este ciclo de máquina
      "accumulator"   → RESULT actual del PE (para inspección)
      null / None     → 0.0
    """
    if src is None:
        return 0.0
    if src == "sequence":
        return xi
    if src.startswith("spatial:"):
        arr = spatial.get(src[8:], np.zeros(1))
        return float(arr[j % len(arr)])
    if src.startswith("const:"):
        return float(src[6:])
    if src.startswith("stage:"):
        return stage_results.get(int(src[6:]), 0.0)
    if src == "accumulator":
        return pe.RESULT
    if src == "zero":  return 0.0
    if src == "one":   return 1.0
    try:
        return float(src)
    except ValueError:
        raise ValueError(f"Fuente de operando no reconocida: {src!r}")


# ──────────────────────────────────────────────────────────────────────────────
# EJECUTOR DE ALGORITMOS
# ──────────────────────────────────────────────────────────────────────────────

class AlgorithmRunner:
    """
    Carga un algoritmo desde su diccionario JSON y lo simula sobre una MMI.

    El modelo es BEHAVIORAL: cada etapa declara explícitamente sus fuentes de
    operandos, lo que reproduce el grafo de flujo de datos de PALOMA sin necesidad
    de modelar las células de interconexión hardware.
    """

    def __init__(self, algo: dict, rows: int, cols: int, N: int, M: int):
        self.algo = algo
        defs = algo.get("defaults", {})
        # Los parámetros CLI tienen prioridad sobre los defaults del JSON
        self.rows = rows if rows is not None else defs.get("rows", 8)
        self.cols = cols if cols is not None else defs.get("cols", 16)
        self.N    = N    if N    is not None else defs.get("N",    200)
        self.M    = M    if M    is not None else defs.get("M",    16)

    def _gen_data(self, seed: int = 42) -> tuple:
        seq_spec = self.algo.get("sequence", {"type": "random_uniform"})
        x = generate_array(seq_spec, self.N, seed)

        # Arrays espaciales declarados directamente
        spatial: Dict[str, np.ndarray] = {}
        for sp in self.algo.get("spatial", []):
            spatial[sp["name"]] = generate_array(sp, self.M, seed)

        # Arrays derivados: calculados con numpy a partir de otros arrays y N/M.
        # Ejemplo: {"name": "FREQ_STEP", "numpy_expr": "FREQ * 2*math.pi / N"}
        # Esto permite expresar parámetros que dependen del tamaño de la ventana
        # sin modificar el motor del simulador.
        ctx = {"np": np, "math": math, "N": self.N, "M": self.M}
        ctx.update(spatial)
        for ds in self.algo.get("derived_spatial", []):
            try:
                result = np.asarray(eval(ds["numpy_expr"], ctx), dtype=float)
                if result.ndim == 0:
                    result = np.full(self.M, float(result))
                spatial[ds["name"]] = result
                ctx[ds["name"]] = result
            except Exception as e:
                print(f"  ⚠  derived_spatial '{ds.get('name')}' falló: {e}")
                spatial[ds["name"]] = np.zeros(self.M)

        return x, spatial

    def _program_mmi(self, mmi: MMI, spatial: Dict[str, np.ndarray]) -> None:
        pipeline = self.algo["pipeline"]
        j_width  = min(self.M, mmi.cols)
        for stage_def in pipeline:
            s   = stage_def["stage"]
            opc = Opcode[stage_def["instruction"]]
            for j in range(j_width):
                p = mmi.pe(s, j)
                p.INSTRUCT = opc
                p.OPERAND  = float(stage_def.get("operand", 0.0))

                # INITVAL escalar: mismo valor para todos los canales
                p.INITVAL = float(stage_def.get("init_val", 0.0))

                # preload_result: carga INITVAL desde un array espacial (por canal).
                # Útil para acumuladores de fase donde cada canal arranca en φ_j.
                # Tiene prioridad sobre init_val si ambos están definidos.
                pr = stage_def.get("preload_result")
                if pr:
                    arr = spatial.get(pr, np.zeros(self.M))
                    p.INITVAL = float(arr[j % len(arr)])

                # En el modelo behavioral los operandos se resuelven directamente,
                # así que no necesitamos stagger: todos los PEs ejecutan cada ciclo.
                p.INITCNT = 0
                p.MAXCNT  = 1

                preload_rega = stage_def.get("preload_rega")
                if preload_rega:
                    p.REGA = resolve_source(preload_rega, 0.0, spatial, j, {}, p)

                p.restart()

    def _simulate(self, mmi: MMI, x: np.ndarray, spatial: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Bucle principal de simulación del pipeline de dataflow.

        Por cada ciclo de máquina, y por cada columna j activa, se resuelven
        los operandos de cada etapa y se llama a PE.execute() en orden de
        dependencia. Los resultados de ciclo anterior fluyen al siguiente
        mediante stage_results[s].
        """
        pipeline  = self.algo["pipeline"]
        n_stages  = len(pipeline)
        j_width   = min(self.M, mmi.cols)
        out_cfg   = self.algo.get("output", {})
        out_stage = out_cfg.get("stage", pipeline[-1]["stage"])
        accum_pe  = out_cfg.get("accumulate_in_pe", False)
        y_accum   = np.zeros(j_width)

        # En el modelo behavioral todas las etapas se resuelven en el mismo ciclo
        # (no hay latencia física de transmisión entre PEs). Basta con N ciclos:
        # no se necesitan ciclos extra de vaciado de pipeline.
        for i in range(self.N):
            xi      = float(x[i])
            max_lat = 1

            for j in range(j_width):
                stage_results: Dict[int, float] = {}

                for stage_def in pipeline:
                    s  = stage_def["stage"]
                    pe = mmi.pe(s, j)
                    pe.REGA = resolve_source(stage_def.get("rega_source"), xi, spatial, j, stage_results, pe)
                    pe.REGB = resolve_source(stage_def.get("regb_source"), xi, spatial, j, stage_results, pe)
                    result  = pe.execute()
                    stage_results[s] = result
                    max_lat = max(max_lat, LATENCY.get(pe.INSTRUCT, 1) if pe.INSTRUCT else 1)
                    pe.tick_counter()

                # Acumulación externa (para etapas de paso, no MULA)
                if not accum_pe:
                    y_accum[j] += stage_results.get(out_stage, 0.0)

            mmi.tick(max_lat)

        if accum_pe:
            return np.array([mmi.pe(out_stage, j).RESULT for j in range(j_width)])
        return y_accum

    def _postprocess(self, y: np.ndarray) -> np.ndarray:
        pp = self.algo.get("output", {}).get("postprocess")
        if pp is None or pp == "identity":    return y
        elif pp == "sqrt":                    return np.sqrt(np.abs(y))
        elif pp == "divide_N":                return y / self.N
        elif pp == "abs":                     return np.abs(y)
        elif pp == "abs_divide_N":            return np.abs(y) / self.N
        elif pp == "negate":                  return -y
        elif pp == "sqrt_divide_N":           return np.sqrt(np.abs(y) / max(1, self.N))
        # 10·log10(acc/N): divide por N ANTES del logaritmo → dBW correctos
        elif pp == "divide_N_then_dB":
            return 10.0 * np.log10(np.maximum(np.abs(y) / max(1, self.N), 1e-12))
        # log10 directo sin dividir (útil si el acumulador ya es una media)
        elif pp == "log10_scale10":
            return 10.0 * np.log10(np.maximum(np.abs(y), 1e-12))
        else:
            raise ValueError(f"Postprocesado desconocido: {pp!r}")

    def _reference(self, x: np.ndarray, spatial: Dict[str, np.ndarray]) -> np.ndarray:
        ref_cfg = self.algo.get("reference", {})
        expr    = ref_cfg.get("numpy_expr", "np.zeros(M)")
        ctx = {"np": np, "math": math, "x": x, "N": self.N, "M": self.M}
        ctx.update(spatial)
        try:
            return np.asarray(eval(expr, ctx), dtype=float)
        except Exception as e:
            print(f"  ⚠  Referencia numpy falló: {e}")
            return np.zeros(self.M)

    def run(self, seed: int = 42, verbose: bool = True) -> dict:
        """Genera datos → programa MMI → simula → compara con referencia numpy."""
        meta = self.algo.get("metadata", {})
        name = meta.get("name", "Algoritmo sin nombre")

        if verbose:
            print(f"\n  Algoritmo : {name}")
            print(f"  Malla     : {self.rows}×{self.cols} = {self.rows*self.cols} PEs")
            print(f"  Secuencia : N={self.N}  Columnas activas: M={self.M}")

        x, spatial = self._gen_data(seed)
        mmi = MMI(self.rows, self.cols)
        mmi.reset()
        self._program_mmi(mmi, spatial)

        t0       = time.perf_counter()
        y_paloma = self._simulate(mmi, x, spatial)
        t_sim    = time.perf_counter() - t0
        y_paloma = self._postprocess(y_paloma)

        t0    = time.perf_counter()
        y_ref = self._reference(x, spatial)
        t_ref = time.perf_counter() - t0

        j_wide = min(len(y_paloma), len(y_ref))
        denom  = np.abs(y_ref[:j_wide]) + 1e-10
        error  = float(np.mean(np.abs(y_paloma[:j_wide] - y_ref[:j_wide]) / denom) * 100.0)

        pipeline = self.algo["pipeline"]
        n_stages = len(pipeline)
        j_width  = min(self.M, self.cols)

        if verbose:
            print(f"  Etapas    : {n_stages}  |  Ciclos máquina: {mmi.machine_cycles:,}")
            print(f"  T. HW est.: {mmi.elapsed_ns:.1f} ns  (@66 MHz)")
            print(f"  Pico HW   : {mmi.peak_gflops:.4f} GFLOPS")
            print(f"  Error ref : {error:.4f} %")

        return {
            "name": name, "rows": self.rows, "cols": self.cols,
            "N": self.N, "M": self.M, "num_pes": mmi.num_pes,
            "n_stages": n_stages, "j_width": j_width,
            "x": x, "spatial": spatial,
            "y_paloma": y_paloma, "y_ref": y_ref,
            "error_rel_pct": error,
            "t_sim_ms": t_sim * 1e3, "t_ref_ms": t_ref * 1e3,
            "machine_cycles": mmi.machine_cycles,
            "clock_cycles": mmi.clock_cycles,
            "estimated_ns": mmi.elapsed_ns,
            "peak_gflops": mmi.peak_gflops,
            "mmi": mmi, "algo": self.algo,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIONES HISTÓRICAS Y ESCALABILIDAD
# ──────────────────────────────────────────────────────────────────────────────

def historical_configurations() -> List[dict]:
    configs = [
        {"name": "Monoprocesador\n(ref. 1998)",  "note": "Pentium II @400MHz",
         "num_pes": 1,         "gflops": CPU_REF_GFLOPS},
        {"name": "Tarjeta MMI 1K",               "note": "16 chips × 64 PEs → PCI",
         "num_pes": 1_024,     "gflops": 1_024      * MFLOPS_PER_PE / 1000},
        {"name": "Equipo 64K",                   "note": "4 placas 19\" × 16K PEs",
         "num_pes": 65_536,    "gflops": 65_536     * MFLOPS_PER_PE / 1000},
        {"name": "PLM 1M",                       "note": "64 PCBs 19\" → cabina 2m",
         "num_pes": 1_048_576, "gflops": 1_048_576  * MFLOPS_PER_PE / 1000},
    ]
    for c in configs:
        c["speedup"] = c["gflops"] / CPU_REF_GFLOPS
    return configs


def scalability_analysis() -> List[dict]:
    data = []
    for s in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        g = s * s * MFLOPS_PER_PE / 1000
        data.append({"label": f"{s}×{s}", "s": s, "num_pes": s*s,
                      "gflops": g, "speedup": g / CPU_REF_GFLOPS})
    return data


# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def build_dashboard(bench: dict, configs: List[dict], scale: List[dict], out_path: str) -> None:
    BG    = "#1A1A2E"; PBG   = "#16213E"; TXT   = "#E0E0E0"
    GREEN = "#06D6A0"; ORANGE= "#FFB703"; GOLD  = "#E94560"; BLUE  = "#533483"

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"PROYECTO PALOMA  ·  Parallel Logic Machine  ·  {bench['name']}",
        fontsize=16, fontweight="bold", color="white", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.36)

    def ax_style(ax, title):
        ax.set_facecolor(PBG)
        ax.set_title(title, color=TXT, fontsize=9, fontweight="bold", pad=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        ax.tick_params(colors=TXT, labelsize=7.5)
        ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        ax.grid(alpha=0.15, color="#888888")

    # ── 1: Mapa de calor de actividad ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mmi  = bench["mmi"]
    heat = np.array([[mmi.grid[r][c].exec_count for c in range(mmi.cols)] for r in range(mmi.rows)])
    im = ax1.imshow(heat, cmap="plasma", interpolation="nearest", aspect="auto")
    ax_style(ax1, f"Actividad PEs — malla {mmi.rows}×{mmi.cols}")
    ax1.set_xlabel("Columna j"); ax1.set_ylabel("Fila (etapa pipeline)")
    cb = plt.colorbar(im, ax=ax1)
    cb.ax.yaxis.set_tick_params(colors=TXT, labelsize=7)
    cb.set_label("Ejecuciones", color=TXT, fontsize=7)

    # ── 2: Comparación PALOMA vs referencia ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    j_show = min(bench["j_width"], 16)
    ax2.plot(range(j_show), bench["y_ref"][:j_show],    "o-", label="Referencia (numpy)",
             color="#64B5F6", linewidth=1.5, markersize=4)
    ax2.plot(range(j_show), bench["y_paloma"][:j_show], "s--", label="PALOMA (sim.)",
             color=GOLD, linewidth=1.5, markersize=4, alpha=0.85)
    ax_style(ax2, f"Benchmark — error: {bench['error_rel_pct']:.4f} %")
    ax2.set_xlabel("Columna j"); ax2.set_ylabel("Resultado")
    ax2.legend(fontsize=7.5, facecolor=PBG, labelcolor=TXT)

    # ── 3: Escalabilidad log-log ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    pes = [d["num_pes"] for d in scale]; gfl = [d["gflops"] for d in scale]
    ax3.loglog(pes, gfl, "o-", color=GREEN, linewidth=2, markersize=5, label="PALOMA")
    ax3.loglog(pes, [p * MFLOPS_PER_PE / 1000 for p in pes], "--",
               color="#AAAAAA", alpha=0.5, linewidth=1, label="Lineal ideal")
    ax_style(ax3, "Escalabilidad Lineal  (log-log)")
    ax3.set_xlabel("Número de PEs"); ax3.set_ylabel("GFLOPS")
    ax3.legend(fontsize=7, facecolor=PBG, labelcolor=TXT)

    # ── 4: Gantt del ciclo de máquina ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PBG)
    fases = [("Fase 1: Ejecución", 0, 32, GREEN), ("Fase 2: Transmisión serie", 32, 33, ORANGE),
             ("Fase 3: Escritura RAM", 0, 33, BLUE), ("Fase 4: Lectura RAM", 0, 33, GOLD)]
    for i, (lbl, off, dur, col) in enumerate(fases):
        ax4.broken_barh([(off, dur)], (i * 1.3, 1.0), facecolors=col, alpha=0.8, edgecolor="#555")
        ax4.text(off + dur / 2, i * 1.3 + 0.5, lbl, ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold")
    ax4.axvline(x=32, color="red", linestyle="--", alpha=0.7, linewidth=1.2)
    ax4.set_xlim(0, 72); ax4.set_ylim(-0.3, 5.5)
    ax4.set_xlabel("Ciclos de reloj"); ax4.set_yticks([])
    ax_style(ax4, "Ciclo de Máquina (4 fases → 2 solapadas)")

    # ── 5: Tabla de métricas ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PBG); ax5.axis("off")
    pipeline = bench["algo"].get("pipeline", [])
    opcodes  = [Opcode[s["instruction"]].value for s in pipeline]
    rows_data = [
        ["PEs en la MMI",             f"{bench['num_pes']:,}"],
        ["Etapas del pipeline",        f"{bench['n_stages']}  →  {', '.join(opcodes)}"],
        ["Ciclos de máquina",          f"{bench['machine_cycles']:,}"],
        ["Ciclos de reloj estimados",  f"{bench['clock_cycles']:,}"],
        ["Tiempo HW (@66 MHz)",        f"{bench['estimated_ns']:.1f} ns"],
        ["Potencia pico",              f"{bench['peak_gflops']:.4f} GFLOPS"],
        ["Error vs referencia",        f"{bench['error_rel_pct']:.4f} %"],
    ]
    tbl = ax5.table(cellText=rows_data, colLabels=["Métrica", "Valor"],
                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#37474F"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1E2A3A" if ri % 2 == 0 else "#243040")
            cell.set_text_props(color=TXT)
        cell.set_edgecolor("#444466")
    ax5.set_title("Métricas de Simulación", color=TXT, fontsize=9, fontweight="bold", pad=8)

    # ── 6: Potencia por configuración ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    names  = [c["name"] for c in configs]; gflops = [c["gflops"] for c in configs]
    bars   = ax6.bar(range(len(names)), gflops,
                     color=["#607D8B", GREEN, ORANGE, GOLD], alpha=0.88, edgecolor="#888", linewidth=0.5)
    ax6.set_yscale("log"); ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, fontsize=7, color=TXT)
    ax6.set_ylabel("GFLOPS  (escala log)")
    for bar, v in zip(bars, gflops):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.6,
                 f"{v:.0f} G" if v < 1000 else f"{v/1000:.1f} T",
                 ha="center", va="bottom", fontsize=8, color=TXT)
    ax_style(ax6, "Potencia pico por Configuración")

    # ── 7: Latencias del repertorio ────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    lat_show = {"ADD":1,"MUL":4,"DIV":32,"MULA":5,"SQRT":8,"LOG10":16,
                "SIN":20,"POW":24,"DEL2":1,"LD/ST":33}
    ax7.barh(list(lat_show.keys()), list(lat_show.values()),
             color=BLUE, alpha=0.8, edgecolor="#888")
    ax7.set_xlabel("Ciclos de reloj (CLKINS)")
    ax_style(ax7, "Latencias del Repertorio (64 instrucciones)")
    for i, (k, v) in enumerate(lat_show.items()):
        ax7.text(v + 0.3, i, str(v), va="center", fontsize=8, color=TXT)

    # ── 8: Pipeline temporal ──────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_facecolor(PBG)
    n_s    = bench["n_stages"]
    n_show = min(16, bench["N"] + n_s)
    cmap_p = plt.cm.YlGnBu
    for s_idx in range(min(n_s, bench["rows"])):
        for i in range(n_show - s_idx):
            ax8.broken_barh([(i + s_idx, 0.85)], (s_idx * 1.3, 1.0),
                            facecolors=cmap_p(0.2 + 0.12 * s_idx), edgecolor="#333", linewidth=0.4)
    ax8.set_xlim(0, n_show); ax8.set_ylim(-0.3, min(n_s, bench["rows"]) * 1.3 + 0.3)
    stg_labels = [p["instruction"] for p in pipeline[:min(n_s, bench["rows"])]]
    ax8.set_yticks([s * 1.3 + 0.5 for s in range(len(stg_labels))])
    ax8.set_yticklabels(stg_labels, fontsize=8, color=TXT)
    ax8.set_xlabel("Ciclo de máquina")
    ax_style(ax8, "Pipeline Temporal (stagger por etapa)")

    # ── 9: Speedup vs CPU ─────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    speedups  = [c["speedup"] for c in configs]
    cfg_names = [c["name"] for c in configs]
    ax9.barh(range(len(cfg_names)), speedups,
             color=["#607D8B", GREEN, ORANGE, GOLD], alpha=0.85, edgecolor="#888", linewidth=0.5)
    ax9.set_yticks(range(len(cfg_names)))
    ax9.set_yticklabels(cfg_names, fontsize=7.5, color=TXT)
    ax9.set_xscale("log"); ax9.set_xlabel("Speedup  (escala log)")
    ax9.axvline(x=1, color="red", linestyle="--", alpha=0.6)
    for i, sp in enumerate(speedups):
        ax9.text(sp * 1.2, i, f"{sp:.0f}×", va="center", fontsize=8, color=TXT)
    ax_style(ax9, "Aceleración vs Pentium II  (~1998)")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  ✓  Dashboard guardado → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITMO INTEGRADO — benchmark canónico de la memoria técnica (pág. 11-13)
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_POLYNOMIAL = {
    "metadata": {
        "name": "Polinomio Vectorial (benchmark original)",
        "description": "y_j = Σ_{i=0}^{N-1} (A_j·x_i² + B_j·x_i + C_j)",
        "reference_formula": "y_j = Σ(A·x² + B·x + C)",
        "author": "Prototec S.L. — Memoria Técnica, pág. 11-13",
    },
    "defaults": {"rows": 8, "cols": 16, "N": 200, "M": 16},
    "sequence": {"type": "random_uniform", "range": [-5.0, 5.0], "seed": 42},
    "spatial": [
        {"name": "A", "type": "random_uniform", "range": [0.1, 3.0],  "seed": 10},
        {"name": "B", "type": "random_uniform", "range": [-2.0, 2.0], "seed": 11},
        {"name": "C", "type": "random_uniform", "range": [-1.0, 1.0], "seed": 12},
    ],
    "pipeline": [
        {"stage": 0, "instruction": "MUL",  "description": "x² = x·x",
         "rega_source": "sequence", "regb_source": "sequence", "init_cnt_offset": 0},
        {"stage": 1, "instruction": "MUL",  "description": "A·x²",
         "rega_source": "spatial:A", "regb_source": "stage:0", "init_cnt_offset": 1},
        {"stage": 2, "instruction": "MUL",  "description": "B·x",
         "rega_source": "spatial:B", "regb_source": "sequence", "init_cnt_offset": 2},
        {"stage": 3, "instruction": "ADD",  "description": "A·x² + B·x",
         "rega_source": "stage:1",  "regb_source": "stage:2",  "init_cnt_offset": 3},
        {"stage": 4, "instruction": "ADD",  "description": "+ C",
         "rega_source": "stage:3",  "regb_source": "spatial:C","init_cnt_offset": 4},
        {"stage": 5, "instruction": "MULA", "description": "y_j += resultado",
         "rega_source": "const:1.0","regb_source": "stage:4",  "init_cnt_offset": 5,
         "init_val": 0.0},
    ],
    "output": {"stage": 5, "accumulate_in_pe": True},
    "reference": {
        "numpy_expr": "np.array([np.sum(A[j]*x**2 + B[j]*x + C[j]) for j in range(M)])",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="paloma_sim.py",
        description=(
            "Simulador del Proyecto PALOMA — Parallel Logic Machine (Prototec S.L., ~1998).\n"
            "Modela la arquitectura MMI: malla masivamente paralela de procesadores\n"
            "elementales con comunicación local y modelo de flujo de datos síncrono."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python paloma_sim.py\n"
            "  python paloma_sim.py --rows 16 --cols 16 --N 500 --M 16\n"
            "  python paloma_sim.py --algorithm algorithms/rms.json\n"
            "  python paloma_sim.py --algorithm algorithms/fft_power.json --N 1024\n"
            "  python paloma_sim.py --list\n"
        ),
    )
    p.add_argument("--algorithm", "-a", metavar="ARCHIVO.json",
                   help="Algoritmo JSON a ejecutar. Sin este flag: benchmark del polinomio.")
    p.add_argument("--rows",    "-r", type=int, default=None, help="Filas de la malla (etapas).")
    p.add_argument("--cols",    "-c", type=int, default=None, help="Columnas de la malla.")
    p.add_argument("--N",       "-n", type=int, default=None, help="Longitud de la secuencia temporal.")
    p.add_argument("--M",       "-m", type=int, default=None, help="Número de canales espaciales.")
    p.add_argument("--seed",    "-s", type=int, default=42,   help="Semilla RNG. Default: 42.")
    p.add_argument("--output",  "-o", metavar="PNG",          default="paloma_simulation.png",
                   help="Ruta del dashboard de salida. Default: paloma_simulation.png.")
    p.add_argument("--no-dashboard", action="store_true",     help="Omitir el dashboard visual.")
    p.add_argument("--list",    "-l", action="store_true",    help="Listar algoritmos disponibles.")
    p.add_argument("--quiet",   "-q", action="store_true",    help="Reducir salida por consola.")
    return p.parse_args()


def list_algorithms() -> None:
    print("\n  Algoritmos disponibles en ./algorithms/\n")
    if not ALG_DIR.exists():
        print("  (directorio algorithms/ no encontrado)"); return
    files = sorted(ALG_DIR.glob("*.json"))
    if not files:
        print("  (no hay archivos .json en algorithms/)"); return
    for f in files:
        try:
            with open(f) as fh: data = json.load(fh)
            meta    = data.get("metadata", {})
            name    = meta.get("name", f.stem)
            formula = meta.get("reference_formula", "")
            defs    = data.get("defaults", {})
            print(f"  {f.name:<38}  {name}")
            if formula: print(f"  {'':38}  → {formula}")
            print(f"  {'':38}  defaults: rows={defs.get('rows','?')} cols={defs.get('cols','?')} "
                  f"N={defs.get('N','?')} M={defs.get('M','?')}\n")
        except Exception as e:
            print(f"  {f.name}: error al leer ({e})\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.list:
        list_algorithms(); sys.exit(0)

    print()
    print("═" * 65)
    print("  PROYECTO PALOMA — Parallel Logic Machine")
    print("  Simulador Python v2.0  ·  Prototec S.L. (~1998)")
    print("═" * 65)

    if args.algorithm:
        alg_path = Path(args.algorithm)
        if not alg_path.exists():
            alg_path = ALG_DIR / args.algorithm
        if not alg_path.exists():
            print(f"\n  ✗  Archivo no encontrado: {args.algorithm}")
            print(f"     Usa --list para ver los algoritmos disponibles.")
            sys.exit(1)
        with open(alg_path) as fh:
            algo_def = json.load(fh)
        print(f"\n  Cargando algoritmo: {alg_path.name}")
    else:
        algo_def = BUILTIN_POLYNOMIAL
        print("\n  Usando benchmark integrado: polinomio vectorial (memoria técnica)")

    runner = AlgorithmRunner(algo_def, rows=args.rows, cols=args.cols, N=args.N, M=args.M)
    bench  = runner.run(seed=args.seed, verbose=not args.quiet)

    configs = historical_configurations()
    scale   = scalability_analysis()

    if not args.quiet:
        print("\n[Configuraciones históricas]")
        print(f"\n  {'Configuración':<22} {'PEs':>12} {'GFLOPS':>10} {'Speedup':>10}  Notas")
        print("  " + "─" * 68)
        for c in configs:
            print(f"  {c['name'].replace(chr(10),' '):<22} {c['num_pes']:>12,} "
                  f"{c['gflops']:>10.1f} {c['speedup']:>9.0f}×  {c['note']}")

        print("\n[Escalabilidad]")
        print(f"\n  {'Malla':<8} {'PEs':>10} {'GFLOPS':>10} {'Speedup':>10}")
        print("  " + "─" * 42)
        for d in scale:
            print(f"  {d['label']:<8} {d['num_pes']:>10,} {d['gflops']:>10.2f} {d['speedup']:>10.0f}×")

    if not args.no_dashboard:
        if not args.quiet: print("\n  Generando dashboard visual...")
        build_dashboard(bench, configs, scale, args.output)

    print()
    print("═" * 65)
    print("  Simulación completada.")
    print("═" * 65)
    print()


if __name__ == "__main__":
    main()
