"""Máquina de estados para presença e simulação de bloqueio."""

from collections.abc import Callable
from enum import Enum, auto
import math
import time


class EstadoPresenca(Enum):
    """Estados possíveis do controlador de presença."""

    INICIALIZANDO = auto()
    PRESENTE = auto()
    AUSENCIA_TEMPORARIA = auto()
    BLOQUEIO_PENDENTE = auto()
    BLOQUEADO = auto()
    ERRO = auto()


class ControladorPresenca:
    """Controla uma ausência contínua com relógio monotônico injetável."""

    def __init__(self, tempo_limite: float = 10.0,
                 relogio: Callable[[], float] = time.monotonic) -> None:
        """Inicializa o controlador sem presumir presença ou ausência."""
        if not math.isfinite(tempo_limite) or tempo_limite <= 0:
            raise ValueError("O tempo limite deve ser positivo e finito.")
        self.tempo_limite = tempo_limite
        self._relogio = relogio
        self.reinicializar()

    def reinicializar(self) -> None:
        """Reinicia explicitamente o controle, inclusive após bloqueio ou erro."""
        self.estado = EstadoPresenca.INICIALIZANDO
        self.inicio_ausencia: float | None = None
        self.instante_solicitacao: float | None = None
        self.erro: str | None = None

    @property
    def tempo_ausencia(self) -> float:
        """Retorna a duração da ausência até a solicitação ou o instante atual."""
        if self.inicio_ausencia is None:
            return 0.0
        fim = self.instante_solicitacao if self.instante_solicitacao is not None else self._relogio()
        return max(0.0, fim - self.inicio_ausencia)

    @property
    def tempo_restante(self) -> float:
        """Retorna os segundos restantes até o limite, sem valores negativos."""
        return max(0.0, self.tempo_limite - self.tempo_ausencia)

    @property
    def solicitar_bloqueio(self) -> bool:
        """Indica que há uma solicitação ainda não confirmada."""
        return self.estado == EstadoPresenca.BLOQUEIO_PENDENTE

    def atualizar(self, presenca_valida: bool) -> EstadoPresenca:
        """Processa uma inferência bem-sucedida, preservando estados terminais."""
        if self.estado in (EstadoPresenca.BLOQUEIO_PENDENTE, EstadoPresenca.BLOQUEADO, EstadoPresenca.ERRO):
            return self.estado
        agora = self._relogio()
        if presenca_valida:
            self.inicio_ausencia = None
            self.estado = EstadoPresenca.PRESENTE
        else:
            if self.inicio_ausencia is None:
                self.inicio_ausencia = agora
            if agora - self.inicio_ausencia >= self.tempo_limite:
                self.instante_solicitacao = agora
                self.estado = EstadoPresenca.BLOQUEIO_PENDENTE
            else:
                self.estado = EstadoPresenca.AUSENCIA_TEMPORARIA
        return self.estado

    def confirmar_bloqueio(self) -> None:
        """Confirma o processamento da solicitação, sem bloquear o sistema."""
        if not self.solicitar_bloqueio:
            raise RuntimeError("Não há solicitação de bloqueio pendente.")
        self.estado = EstadoPresenca.BLOQUEADO

    def registrar_erro(self, erro: str) -> None:
        """Registra uma falha sem interpretá-la como ausência normal."""
        self.erro = erro
        self.estado = EstadoPresenca.ERRO
        self.inicio_ausencia = None
