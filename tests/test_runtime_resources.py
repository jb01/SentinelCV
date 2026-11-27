import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

import main
from src.config import DEFAULTS, validar_config
from src.presence import ControladorPresenca, EstadoPresenca
from src.resources import Recursos
from src.runtime import Limitador


def test_limitador_sem_espera_ocupada():
    agora = [0.0]
    esperas = []
    def dormir(tempo):
        esperas.append(tempo)
        agora[0] += tempo
    limite = Limitador(2, relogio=lambda: agora[0], dormir=dormir)
    limite.aguardar()
    limite.aguardar()
    assert esperas == [0.5]
    agora[0] += 1
    limite.aguardar()
    assert esperas == [0.5]


def test_temporizador_usa_tempo_real():
    agora = [0.0]
    controlador = ControladorPresenca(10, relogio=lambda: agora[0])
    controlador.atualizar(False)
    agora[0] = 9.9
    controlador.atualizar(False)
    assert controlador.estado == EstadoPresenca.AUSENCIA_TEMPORARIA
    agora[0] = 10.1
    controlador.atualizar(False)
    assert controlador.estado == EstadoPresenca.BLOQUEIO_PENDENTE


@pytest.mark.parametrize('chave,valor', [('inference_fps', 0), ('inference_fps', float('inf')),
                                         ('cpu_threads', 0), ('headless', 'sim'), ('resource_interval', 0)])
def test_configuracoes_invalidas(chave, valor):
    config = copy.deepcopy(DEFAULTS)
    config['runtime'][chave] = valor
    with pytest.raises(ValueError):
        validar_config(config)


def test_coleta_exporta_metricas(tmp_path):
    processo = Mock()
    processo.cpu_percent.side_effect = [0, 80, 120]
    processo.memory_info.side_effect = [SimpleNamespace(rss=10485760), SimpleNamespace(rss=12582912)]
    processo.memory_percent.return_value = 1
    processo.num_threads.return_value = 4
    psutil = SimpleNamespace(Process=lambda: processo, cpu_count=lambda logical: 4)
    recursos = Recursos(1, {'runtime': {}}, psutil)
    recursos.atualizar(5, 10)
    recursos.coletar()
    recursos.coletar()
    csv, resumo = recursos.exportar(tmp_path, 'teste')
    dados = json.loads(resumo.read_text())
    assert csv.exists()
    assert dados['cpu_processo_media_pct'] == 100
    assert dados['cpu_normalizada_media_pct'] == 25
    assert dados['variacao_ram_mib'] == 2
    assert dados['fps_medio'] == 5


def test_headless_ctrl_c_libera_camera(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['main.py', '--headless'])
    camera = Mock()
    camera.parametros.return_value = (640, 480, 30, 'MJPG')
    camera.ler.side_effect = KeyboardInterrupt
    with patch('main.Camera', return_value=camera), patch('main.Detector'), \
         patch('main.BloqueadorSessao') as bloqueador, patch('main.cv2.imshow') as imshow, \
         patch('main.renderizar') as renderizar, patch('main.cv2.destroyAllWindows') as destruir:
        assert main.main() == 0
    camera.liberar.assert_called_once()
    bloqueador.return_value.bloquear.assert_not_called()
    imshow.assert_not_called()
    renderizar.assert_not_called()
    destruir.assert_not_called()


def test_headless_processa_sem_renderizar(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['main.py', '--headless'])
    camera = Mock()
    camera.parametros.return_value = (640, 480, 30, 'MJPG')
    camera.ler.side_effect = [np.zeros((480, 640, 3), dtype=np.uint8), KeyboardInterrupt()]
    detector = Mock()
    detector.detectar.return_value = []
    detector.latencia_ms = 3
    with patch('main.Camera', return_value=camera), patch('main.Detector', return_value=detector), \
         patch('main.renderizar') as renderizar, patch('main.cv2.imshow') as imshow:
        assert main.main() == 0
    renderizar.assert_not_called()
    imshow.assert_not_called()
    camera.liberar.assert_called_once()


def test_benchmark_recursos_bloqueio_incompativel():
    with pytest.raises(SystemExit):
        main.interpretar_argumentos(['--resource-benchmark', '--lock-enabled'])


def test_captura_entrega_quadro_mais_recente():
    import threading
    from src.runtime import CapturaRecente

    terminou = threading.Event()
    liberar = threading.Event()
    class CameraFalsa:
        def __init__(self):
            self.numero = 0
        def ler(self):
            self.numero += 1
            if self.numero <= 3:
                return self.numero
            terminou.set()
            liberar.wait(2)
            raise RuntimeError('fim')
        def liberar(self):
            liberar.set()
    camera = CameraFalsa()
    captura = CapturaRecente(camera)
    assert terminou.wait(1)
    assert captura.ler() == 3
    liberar.set()
    captura.liberar()


def test_opcoes_runtime_prevalecem_sobre_yaml(tmp_path):
    arquivo = tmp_path / 'config.yaml'
    arquivo.write_text('runtime:\n  inference_fps: 2\n  cpu_threads: 3\n  headless: false\n')
    _, config = main.interpretar_argumentos(['--config', str(arquivo), '--inference-fps', '5',
                                            '--cpu-threads', '1', '--headless'])
    assert config['runtime']['inference_fps'] == 5
    assert config['runtime']['cpu_threads'] == 1
    assert config['runtime']['headless'] is True
