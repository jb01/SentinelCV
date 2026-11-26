"""Testes do detector sem câmera nem pesos reais."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.detector import Deteccao, Detector


@pytest.fixture
def deteccao() -> Deteccao:
    """Fornece uma caixa de dimensões conhecidas."""
    return Deteccao(10.0, 20.0, 110.0, 70.0, 0.9, 0)


def test_geometria(deteccao: Deteccao) -> None:
    assert deteccao.largura == 100
    assert deteccao.altura == 50
    assert deteccao.area == 5000
    assert deteccao.centro == (60, 45)
    assert deteccao.area_relativa(200, 100) == 0.25
    assert deteccao.coordenadas_normalizadas(200, 100) == (0.3, 0.45, 0.5, 0.5)


@pytest.mark.parametrize("dimensoes", [(0, 100), (200, 0), (-1, 100)])
def test_dimensoes_invalidas(deteccao: Deteccao, dimensoes: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        deteccao.area_relativa(*dimensoes)
    with pytest.raises(ValueError):
        deteccao.coordenadas_normalizadas(*dimensoes)


def test_sem_deteccoes() -> None:
    with patch("src.detector.YOLO") as modelo:
        caixas = SimpleNamespace(xyxy=[], conf=[], cls=[])
        modelo.return_value.predict.return_value = [SimpleNamespace(boxes=caixas)]
        detector = Detector()
        quadro = np.zeros((100, 200, 3), dtype=np.uint8)
        assert detector.detectar(quadro) == []
        modelo.assert_called_once_with("yolo26n.pt")
        modelo.return_value.predict.assert_called_once_with(
            source=quadro, device="cpu", imgsz=320, conf=0.5, classes=[0], verbose=False,
        )
        assert detector.latencia_ms >= 0


def test_deteccao_e_falha_propagada() -> None:
    with patch("src.detector.YOLO") as modelo:
        caixas = SimpleNamespace(xyxy=[[10, 20, 110, 70]], conf=[0.9], cls=[0])
        modelo.return_value.predict.return_value = [SimpleNamespace(boxes=caixas)]
        detector = Detector()
        quadro = np.zeros((100, 200, 3), dtype=np.uint8)
        assert detector.detectar(quadro) == [Deteccao(10, 20, 110, 70, 0.9, 0)]
        modelo.return_value.predict.side_effect = RuntimeError("falha de inferência")
        with pytest.raises(RuntimeError, match="falha de inferência"):
            detector.detectar(quadro)
