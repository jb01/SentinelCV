"""Solicitação restrita de bloqueio da sessão atual via loginctl."""

from dataclasses import dataclass
import subprocess
import time


@dataclass(frozen=True)
class ResultadoBloqueio:
    """Resultado da solicitação e indicação opcional do estado da sessão."""

    sucesso_solicitacao: bool
    modo_simulado: bool
    latencia_ms: float
    codigo_retorno: int | None
    mensagem: str
    bloqueio_reportado: bool | None = None


class BloqueadorSessao:
    """Executa somente a ação fixa de bloqueio, uma vez por ciclo."""

    def __init__(self, modo_simulado: bool = True) -> None:
        """Configura o modo sem realizar nenhuma ação no sistema."""
        self.modo_simulado = modo_simulado
        self._resultado: ResultadoBloqueio | None = None

    def reinicializar(self) -> None:
        """Permite uma nova solicitação em um ciclo explicitamente reiniciado."""
        self._resultado = None

    def bloquear(self) -> ResultadoBloqueio:
        """Solicita uma vez o bloqueio; repetição devolve o resultado anterior."""
        if self._resultado is not None:
            return self._resultado
        if self.modo_simulado:
            self._resultado = ResultadoBloqueio(True, True, 0.0, None, "Bloqueio simulado; nenhuma ação no sistema.")
            return self._resultado
        inicio = time.perf_counter()
        try:
            processo = subprocess.run(["loginctl", "lock-session"], shell=False, timeout=5,
                                      capture_output=True, text=True, check=False)
        except FileNotFoundError:
            self._resultado = ResultadoBloqueio(False, False, (time.perf_counter() - inicio) * 1000,
                                               None, "Comando loginctl indisponível.")
            return self._resultado
        except subprocess.TimeoutExpired:
            self._resultado = ResultadoBloqueio(False, False, (time.perf_counter() - inicio) * 1000,
                                               None, "Tempo limite da solicitação de bloqueio excedido.")
            return self._resultado
        except OSError:
            self._resultado = ResultadoBloqueio(False, False, (time.perf_counter() - inicio) * 1000,
                                               None, "Falha ao executar loginctl.")
            return self._resultado
        latencia = (time.perf_counter() - inicio) * 1000
        if processo.returncode != 0:
            self._resultado = ResultadoBloqueio(False, False, latencia, processo.returncode,
                                               "Solicitação de bloqueio rejeitada pelo sistema.")
            return self._resultado
        reportado = self._consultar_locked_hint()
        mensagens = {
            True: "Solicitação aceita; LockedHint reporta sessão bloqueada.",
            False: "Solicitação aceita; LockedHint ainda não reporta bloqueio efetivo.",
            None: "Solicitação aceita; não foi possível confirmar o bloqueio via LockedHint.",
        }
        self._resultado = ResultadoBloqueio(True, False, latencia, processo.returncode,
                                           mensagens[reportado], reportado)
        return self._resultado

    @staticmethod
    def _consultar_locked_hint() -> bool | None:
        """Consulta a propriedade da mesma sessão, sem presumir seu estado."""
        try:
            processo = subprocess.run(["loginctl", "show-session", "self", "--property=LockedHint", "--value"],
                                      shell=False, timeout=5, capture_output=True, text=True, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if processo.returncode != 0:
            return None
        return {"yes": True, "no": False}.get(processo.stdout.strip().lower())
