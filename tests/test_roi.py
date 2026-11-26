"""Testa a classificação geométrica sem webcam ou inferência."""

import pytest

from src.detector import Deteccao
from src.roi import FiltroPresenca, ROI


def caixa(x1, y1, x2, y2, classe=0):
    """Cria uma detecção para um quadro de 100 por 100 pixels."""
    return Deteccao(x1, y1, x2, y2, 0.9, classe)


def test_area_e_posicao():
    filtro = FiltroPresenca(min_area=0.15)
    dentro = caixa(25, 25, 75, 75)
    fora = caixa(0, 0, 20, 20)
    pequena = caixa(45, 45, 55, 55)
    limite = caixa(25, 35, 75, 65)
    validas, descartadas = filtro.classificar([dentro, fora, pequena, limite], 100, 100)
    assert validas == [dentro, limite]
    assert descartadas == [fora, pequena]
    assert filtro.motivos_descarte(fora, 100, 100) == ["area_insuficiente", "centro_fora_roi"]
    assert filtro.motivos_descarte(pequena, 100, 100) == ["area_insuficiente"]


@pytest.mark.parametrize("x,y", [(0.15, 0.2), (0.85, 1), (0.15, 1), (0.85, 0.2)])
def test_bordas_inclusivas(x, y):
    filtro = FiltroPresenca(min_area=0)
    deteccao = caixa(x * 100, y * 100, x * 100, y * 100)
    assert filtro.classificar([deteccao], 100, 100) == ([deteccao], [])


def test_vazio_e_classe():
    filtro = FiltroPresenca()
    outra = caixa(0, 0, 100, 100, classe=1)
    assert filtro.classificar([], 100, 100) == ([], [])
    assert filtro.motivos_descarte(outra, 100, 100) == ["classe_nao_person"]


def test_filtro_desabilitado():
    filtro = FiltroPresenca(habilitado=False)
    pessoa = caixa(0, 0, 2, 2)
    outra = caixa(0, 0, 2, 2, classe=1)
    assert filtro.classificar([pessoa, outra], 100, 100) == ([pessoa], [outra])


@pytest.mark.parametrize("coordenadas", [(-0.1, 0, 1, 1), (0, 0, 1.1, 1),
                                             (0.5, 0, 0.5, 1), (0, 1, 1, 0),
                                             (float('nan'), 0, 1, 1)])
def test_roi_invalida(coordenadas):
    with pytest.raises(ValueError):
        ROI(*coordenadas)


@pytest.mark.parametrize("area", [-0.1, 1.1, float('nan')])
def test_area_invalida(area):
    with pytest.raises(ValueError):
        FiltroPresenca(min_area=area)


def test_area_padrao_vinte_por_cento():
    filtro = FiltroPresenca()
    abaixo = caixa(25, 35, 75, 74)
    limite = caixa(25, 30, 75, 70)
    assert filtro.classificar([abaixo, limite], 100, 100) == ([limite], [abaixo])
