"""Testes de bloqueio com subprocessos e câmera simulados."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import subprocess
import sys

import pytest

import main
from src.screen_lock import BloqueadorSessao


def processo(codigo=0, saida=""):
    return subprocess.CompletedProcess([], codigo, stdout=saida, stderr="")


def test_simulacao_sem_subprocesso():
    with patch("src.screen_lock.subprocess.run") as executar:
        resultado = BloqueadorSessao().bloquear()
    assert resultado.sucesso_solicitacao and resultado.modo_simulado
    assert resultado.codigo_retorno is None
    executar.assert_not_called()


def test_real_aceito_e_verificado():
    with patch("src.screen_lock.subprocess.run", side_effect=[processo(), processo(saida="yes\n")]) as executar:
        bloqueador = BloqueadorSessao(False)
        resultado = bloqueador.bloquear()
        assert bloqueador.bloquear() is resultado
    assert resultado.sucesso_solicitacao and not resultado.modo_simulado
    assert resultado.codigo_retorno == 0 and resultado.bloqueio_reportado is True
    assert executar.call_count == 2
    assert executar.call_args_list[0].args == (["loginctl", "lock-session"],)
    assert executar.call_args_list[0].kwargs == {"shell": False, "timeout": 5, "capture_output": True,
                                                   "text": True, "check": False}
    assert executar.call_args_list[1].args == (["loginctl", "show-session", "self", "--property=LockedHint", "--value"],)


@pytest.mark.parametrize("efeito", [processo(1), FileNotFoundError(), subprocess.TimeoutExpired("loginctl", 5), OSError()])
def test_falha_solicitacao_nao_repete(efeito):
    with patch("src.screen_lock.subprocess.run", side_effect=[efeito]) as executar:
        bloqueador = BloqueadorSessao(False)
        resultado = bloqueador.bloquear()
        assert not resultado.sucesso_solicitacao
        assert bloqueador.bloquear() is resultado
        assert executar.call_count == 1


@pytest.mark.parametrize("efeito,esperado", [(processo(saida="no\n"), False), (FileNotFoundError(), None),
                                               (processo(1), None)])
def test_confirmacao_indisponivel_nao_altera_sucesso(efeito, esperado):
    with patch("src.screen_lock.subprocess.run", side_effect=[processo(), efeito]):
        resultado = BloqueadorSessao(False).bloquear()
    assert resultado.sucesso_solicitacao
    assert resultado.bloqueio_reportado is esperado
    assert "confirma" in resultado.mensagem or "ainda não" in resultado.mensagem


def test_incompatibilidade_antes_de_iniciar(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--benchmark", "--lock-enabled"])
    with patch("main.BloqueadorSessao") as bloqueador, pytest.raises(SystemExit) as erro:
        main.main()
    assert erro.value.code == 2
    bloqueador.assert_not_called()


@pytest.mark.parametrize("falha", ["captura", "inferencia"])
@pytest.mark.parametrize("real", [False, True])
def test_falha_camera_ou_inferencia_solicita_uma_vez(monkeypatch, tmp_path, falha, real):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py"] + (["--lock-enabled"] if real else []))
    camera = Mock()
    camera.parametros.return_value = (640, 480, 30, "MJPG")
    detector = Mock()
    if falha == "captura":
        camera.ler.side_effect = RuntimeError("erro da câmera")
    else:
        camera.ler.return_value = object()
        detector.detectar.side_effect = RuntimeError("erro do modelo")
    retorno = SimpleNamespace(sucesso_solicitacao=True, modo_simulado=not real, latencia_ms=1,
                              mensagem="solicitação de teste")
    bloqueador = Mock()
    bloqueador.bloquear.return_value = retorno
    with patch("main.Camera", return_value=camera), patch("main.Detector", return_value=detector), \
         patch("main.BloqueadorSessao", return_value=bloqueador), patch("main.cv2.destroyAllWindows"), \
         patch("main.configurar_log") as log:
        assert main.main() == 1
    bloqueador.bloquear.assert_called_once()
    camera.liberar.assert_called_once()
    assert any(f'falha_{falha}' in str(chamada) for chamada in log.return_value.info.call_args_list)


@pytest.mark.parametrize("sucesso", [True, False])
def test_limite_real_encerra_e_libera_camera(monkeypatch, tmp_path, sucesso):
    import numpy as np

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py", "--lock-enabled", "--absence-time", "0.001"])
    camera = Mock()
    camera.parametros.return_value = (640, 480, 30, "MJPG")
    camera.ler.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = Mock()
    detector.detectar.return_value = []
    detector.latencia_ms = 1.0
    bloqueador = Mock()
    bloqueador.bloquear.return_value = SimpleNamespace(
        sucesso_solicitacao=sucesso, modo_simulado=False, latencia_ms=1,
        mensagem="Solicitação aceita; bloqueio não confirmado" if sucesso else "Falha")
    with patch("main.Camera", return_value=camera), patch("main.Detector", return_value=detector), \
         patch("main.BloqueadorSessao", return_value=bloqueador), patch("main.configurar_log"), \
         patch("main.cv2.imshow"), patch("main.cv2.waitKey", return_value=-1), \
         patch("main.cv2.destroyAllWindows"), patch("main.time.perf_counter", side_effect=iter(range(1000))):
        assert main.main() == (0 if sucesso else 1)
    bloqueador.bloquear.assert_called_once()
    camera.liberar.assert_called_once()
