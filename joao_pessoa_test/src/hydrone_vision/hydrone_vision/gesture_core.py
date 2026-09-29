#!/usr/bin/env python3
"""
gesture_core.py — classificação de gestos por ângulo de braço, para MediaPipe Pose.

Porta de frl_core.py (validado standalone com webcam) para os landmarks do
MediaPipe (33 pontos, normalizados 0..1). A única mudança real é converter
landmarks normalizados para pixels antes de medir ângulos, porque x e y do
MediaPipe são normalizados por largura e altura SEPARADAMENTE — usar (x, y)
normalizado direto distorce o ângulo se a imagem não for quadrada.

Local sugerido: src/hydrone_vision/hydrone_vision/gesture_core.py
"""
import math
from dataclasses import dataclass

# índices MediaPipe Pose
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16

BINS = (("DOWN", 0, 22), ("DIAG_DOWN", 33, 62), ("SIDE", 75, 105),
        ("DIAG_UP", 118, 147), ("UP", 158, 180))
EXT_MIN = 0.75
MIN_VIS = 0.5   # visibility mínima do MediaPipe para confiar no landmark


def _pt(lm, w, h):
    return (lm.x * w, lm.y * h)


def _arm_state(sh, el, wr, vis, w, h):
    if min(vis) < MIN_VIS:
        return "UNK", 0, None
    sh, el, wr = _pt(sh, w, h), _pt(el, w, h), _pt(wr, w, h)
    reach = math.dist(sh, el) + math.dist(el, wr)
    vx, vy = wr[0] - sh[0], wr[1] - sh[1]
    if reach < 1e-6 or math.hypot(vx, vy) / reach < EXT_MIN:
        return "BENT", 0, None
    theta = math.degrees(math.atan2(abs(vx), vy))
    for name, lo, hi in BINS:
        if lo <= theta <= hi:
            return name, (1 if vx > 0 else -1), theta
    return "UNK", 0, theta


def classify(landmarks, img_w: int, img_h: int) -> str:
    """landmarks: results.pose_landmarks.landmark (MediaPipe). Retorna um dos
    nomes de gesto (compatíveis com GESTURE_LABELS / phase3_gesture_node):
    HOVER, STOP, SUBIR, DESCER, DIREITA, ESQUERDA, APROXIMAR, AFASTAR,
    POUSAR, NENHUM.
    """
    l = _arm_state(landmarks[L_SH], landmarks[L_EL], landmarks[L_WR],
                   (landmarks[L_SH].visibility, landmarks[L_EL].visibility,
                    landmarks[L_WR].visibility), img_w, img_h)
    r = _arm_state(landmarks[R_SH], landmarks[R_EL], landmarks[R_WR],
                   (landmarks[R_SH].visibility, landmarks[R_EL].visibility,
                    landmarks[R_WR].visibility), img_w, img_h)
    sl, sr = l[0], r[0]

    def has(a, b):
        return (sl, sr) in ((a, b), (b, a))

    if has("DOWN", "DOWN"):
        return "HOVER"
    if has("SIDE", "SIDE"):
        return "STOP"
    if has("UP", "UP"):
        return "SUBIR"
    if has("DIAG_DOWN", "DIAG_DOWN"):
        return "DESCER"
    if has("UP", "SIDE"):
        return "POUSAR"
    if has("UP", "DOWN"):
        return "AFASTAR"
    if has("DIAG_UP", "DOWN"):
        return "APROXIMAR"
    if has("SIDE", "DOWN"):
        d = l[1] if sl == "SIDE" else r[1]
        # espelhado: o drone encara o operador, então direita da imagem
        # (d>0) é a ESQUERDA do operador. O drone deve seguir o lado que o
        # OPERADOR chama, então invertemos aqui.
        return "ESQUERDA" if d > 0 else "DIREITA"
    return "NENHUM"


@dataclass
class DebounceResult:
    command: str
    changed: bool


HOLD_S = {"HOVER": 0.25, "STOP": 0.25, "POUSAR": 1.2}
DEFAULT_HOLD = 0.5


class Debouncer:
    """Só confirma um gesto após mantido por HOLD_S; volta ao neutro mais rápido."""

    def __init__(self):
        self.active, self.since = "HOVER", 0.0
        self.cand, self.t0 = None, 0.0

    def update(self, gesture: str, t: float) -> DebounceResult:
        g = "HOVER" if gesture == "NENHUM" else gesture
        if g == self.active:
            self.cand = None
            return DebounceResult(self.active, False)
        if g != self.cand:
            self.cand, self.t0 = g, t
        if t - self.t0 >= HOLD_S.get(g, DEFAULT_HOLD):
            self.active, self.since, self.cand = g, t, None
            return DebounceResult(self.active, True)
        return DebounceResult(self.active, False)