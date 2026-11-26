"""Filtragem de presença por área e região de interesse."""

from dataclasses import dataclass
import math

from src.detector import Deteccao


@dataclass(frozen=True)
class ROI:
    """Região retangular em coordenadas normalizadas, com bordas inclusivas."""

    xmin: float = 0.15
    ymin: float = 0.20
    xmax: float = 0.85
    ymax: float = 1.00

    def __post_init__(self) -> None:
        """Valida os limites normalizados da região."""
        if not (0 <= self.xmin < self.xmax <= 1 and 0 <= self.ymin < self.ymax <= 1):
            raise ValueError("A ROI deve ter limites ordenados entre 0 e 1.")

    def contem(self, x: float, y: float) -> bool:
        """Informa se um ponto normalizado está dentro ou na borda da ROI."""
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


class FiltroPresenca:
    """Classifica detecções sem alterar nem remover as caixas originais."""

    def __init__(self, roi: ROI | None = None, min_area: float = 0.20,
                 habilitado: bool = True) -> None:
        """Define a região, a área relativa mínima e o estado da filtragem."""
        if not math.isfinite(min_area) or not 0 <= min_area <= 1:
            raise ValueError("A área mínima deve estar entre 0 e 1.")
        self.roi = roi if roi is not None else ROI()
        self.min_area = min_area
        self.habilitado = habilitado

    def motivos_descarte(self, deteccao: Deteccao, largura: int, altura: int) -> list[str]:
        """Retorna todos os critérios não satisfeitos pela detecção."""
        motivos = []
        if deteccao.classe_id != 0:
            motivos.append("classe_nao_person")
        if self.habilitado:
            if deteccao.area_relativa(largura, altura) < self.min_area:
                motivos.append("area_insuficiente")
            cx, cy, _, _ = deteccao.coordenadas_normalizadas(largura, altura)
            if not self.roi.contem(cx, cy):
                motivos.append("centro_fora_roi")
        return motivos

    def classificar(self, deteccoes: list[Deteccao], largura: int,
                    altura: int) -> tuple[list[Deteccao], list[Deteccao]]:
        """Retorna listas de detecções válidas e descartadas, preservando os objetos."""
        validas, descartadas = [], []
        for deteccao in deteccoes:
            destino = descartadas if self.motivos_descarte(deteccao, largura, altura) else validas
            destino.append(deteccao)
        return validas, descartadas
