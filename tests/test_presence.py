"""Testes da máquina de estados com relógio simulado."""

import pytest

from src.detector import Deteccao
from src.presence import ControladorPresenca, EstadoPresenca as Estado
from src.roi import FiltroPresenca


class Relogio:
    """Relógio monotônico controlado pelo teste."""
    def __init__(self):
        self.agora = 100.0

    def __call__(self):
        return self.agora


@pytest.fixture
def sistema():
    relogio = Relogio()
    return ControladorPresenca(relogio=relogio), relogio


def test_inicializacao_e_presenca_continua(sistema):
    controle, tempo = sistema
    assert controle.estado == Estado.INICIALIZANDO
    assert controle.tempo_ausencia == 0 and controle.tempo_restante == 10
    assert controle.atualizar(True) == Estado.PRESENTE
    tempo.agora += 20
    assert controle.atualizar(True) == Estado.PRESENTE
    assert controle.inicio_ausencia is None


def test_ausencia_inicial_inferior_e_limite_exato(sistema):
    controle, tempo = sistema
    assert controle.atualizar(False) == Estado.AUSENCIA_TEMPORARIA
    assert controle.inicio_ausencia == 100
    tempo.agora = 109.9
    assert controle.atualizar(False) == Estado.AUSENCIA_TEMPORARIA
    assert controle.tempo_restante == pytest.approx(0.1)
    tempo.agora = 110
    assert controle.atualizar(False) == Estado.BLOQUEIO_PENDENTE
    assert controle.solicitar_bloqueio
    assert controle.instante_solicitacao == 110
    assert controle.tempo_ausencia == 10 and controle.tempo_restante == 0


def test_limite_superado_e_solicitacao_unica(sistema):
    controle, tempo = sistema
    controle.atualizar(False)
    tempo.agora = 112
    assert controle.atualizar(False) == Estado.BLOQUEIO_PENDENTE
    assert controle.atualizar(True) == Estado.BLOQUEIO_PENDENTE
    controle.confirmar_bloqueio()
    assert not controle.solicitar_bloqueio
    tempo.agora = 200
    assert controle.atualizar(True) == Estado.BLOQUEADO
    assert controle.atualizar(False) == Estado.BLOQUEADO
    assert controle.tempo_ausencia == 12
    with pytest.raises(RuntimeError):
        controle.confirmar_bloqueio()


def test_retorno_cancelamento_e_reinicio(sistema):
    controle, tempo = sistema
    controle.atualizar(False)
    tempo.agora = 109
    assert controle.atualizar(True) == Estado.PRESENTE
    assert controle.tempo_ausencia == 0 and controle.tempo_restante == 10
    tempo.agora = 111
    controle.atualizar(False)
    assert controle.inicio_ausencia == 111
    tempo.agora = 120
    assert controle.atualizar(False) == Estado.AUSENCIA_TEMPORARIA
    tempo.agora = 121
    assert controle.atualizar(False) == Estado.BLOQUEIO_PENDENTE
    controle.confirmar_bloqueio()
    controle.reinicializar()
    assert controle.estado == Estado.INICIALIZANDO
    assert controle.instante_solicitacao is None and controle.inicio_ausencia is None
    assert controle.atualizar(True) == Estado.PRESENTE


def test_limite_configuravel():
    tempo = Relogio()
    controle = ControladorPresenca(2.5, tempo)
    controle.atualizar(False)
    tempo.agora += 2.5
    assert controle.atualizar(False) == Estado.BLOQUEIO_PENDENTE


def test_erro_exige_reinicializacao(sistema):
    controle, tempo = sistema
    controle.atualizar(False)
    controle.registrar_erro("falha na câmera")
    tempo.agora += 100
    assert controle.estado == Estado.ERRO and controle.erro == "falha na câmera"
    assert controle.atualizar(False) == Estado.ERRO
    assert not controle.solicitar_bloqueio
    controle.reinicializar()
    assert controle.erro is None and controle.estado == Estado.INICIALIZANDO


def test_deteccao_descartada_nao_cancela(sistema):
    controle, tempo = sistema
    filtro = FiltroPresenca()
    controle.atualizar(False)
    tempo.agora += 5
    validas, descartadas = filtro.classificar([Deteccao(0, 0, 10, 10, .9, 0)], 100, 100)
    assert descartadas and not validas
    controle.atualizar(bool(validas))
    assert controle.tempo_ausencia == 5
    tempo.agora += 5
    assert controle.atualizar(False) == Estado.BLOQUEIO_PENDENTE


@pytest.mark.parametrize("limite", [0, -1, float('nan'), float('inf'), -float('inf')])
def test_limite_invalido(limite):
    with pytest.raises(ValueError):
        ControladorPresenca(limite)
