"""Detecção de pessoas com YOLO pré-treinado no COCO.

Referência: https://heltonmaia.com/deeplearning/chapters/ch4/ch4.html
"""

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


@dataclass
class Deteccao:
    """Caixa em coordenadas de pixels do quadro original."""

    x1: float
    y1: float
    x2: float
    y2: float
    confianca: float
    classe_id: int

    @property
    def largura(self) -> float:
        """Retorna a largura da caixa em pixels."""
        return self.x2 - self.x1

    @property
    def altura(self) -> float:
        """Retorna a altura da caixa em pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Retorna a área da caixa em pixels quadrados."""
        return self.largura * self.altura

    @property
    def centro(self) -> tuple[float, float]:
        """Retorna o centro da caixa em pixels."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def area_relativa(self, largura: int, altura: int) -> float:
        """Retorna a proporção da área da caixa em relação ao quadro."""
        if largura <= 0 or altura <= 0:
            raise ValueError("As dimensões do quadro devem ser positivas.")
        return self.area / (largura * altura)

    def coordenadas_normalizadas(self, largura: int, altura: int) -> tuple[float, float, float, float]:
        """Retorna centro, largura e altura normalizados no formato xywhn."""
        if largura <= 0 or altura <= 0:
            raise ValueError("As dimensões do quadro devem ser positivas.")
        cx, cy = self.centro
        return (cx / largura, cy / altura, self.largura / largura, self.altura / altura)


class Detector:
    """Carrega uma vez o modelo e executa inferências na CPU por padrão."""

    def __init__(
        self, modelo: str = "yolo26n.pt", dispositivo: str = "cpu",
        tamanho_inferencia: int = 320, confianca_minima: float = 0.50,
        classes: list[int] | None = None,
    ) -> None:
        """Configura a inferência e carrega os pesos do modelo."""
        if dispositivo != "cpu":
            raise ValueError("Nesta etapa, somente a CPU é permitida.")
        self.modelo = YOLO(modelo)
        self.dispositivo = dispositivo
        self.tamanho_inferencia = tamanho_inferencia
        self.confianca_minima = confianca_minima
        self.classes = [0] if classes is None else list(classes)
        self.latencia_ms = 0.0

    def detectar(self, quadro: NDArray[np.uint8]) -> list[Deteccao]:
        """Retorna as caixas previstas; falhas de inferência são propagadas."""
        inicio = time.perf_counter()
        try:
            resultados = self.modelo.predict(
                source=quadro, device=self.dispositivo, imgsz=self.tamanho_inferencia,
                conf=self.confianca_minima, classes=self.classes, verbose=False,
            )
        finally:
            self.latencia_ms = (time.perf_counter() - inicio) * 1000
        caixas = resultados[0].boxes
        return [
            Deteccao(*(float(v) for v in coordenadas), float(confianca), int(classe))
            for coordenadas, confianca, classe in zip(caixas.xyxy, caixas.conf, caixas.cls)
            if int(classe) == 0
        ]
