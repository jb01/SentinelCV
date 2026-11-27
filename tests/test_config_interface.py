import copy

import numpy as np
import pytest
import yaml

from main import interpretar_argumentos
from src.config import DEFAULTS, carregar_config, validar_config
from src.detector import Deteccao
from src.interface import renderizar
from src.roi import FiltroPresenca


def test_defaults_e_yaml(tmp_path):
    arquivo = tmp_path / 'config.yaml'
    arquivo.write_text('camera:\n  index: 2\n', encoding='utf-8')
    config = carregar_config(arquivo)
    assert config['camera']['index'] == 2
    assert config['benchmark']['duration'] == 30
    assert config['security']['lock_enabled'] is False


@pytest.mark.parametrize('secao,chave,valor', [
    ('camera', 'fps', 0), ('camera', 'codec', 'X'), ('detector', 'device', 'cuda'),
    ('detector', 'confidence', float('nan')), ('detector', 'classes', [1]),
    ('roi', 'xmin', 1), ('roi', 'min_area', -1), ('presence', 'absence_timeout', 0),
    ('benchmark', 'warmup', -1), ('security', 'fail_safe', 'true'),
])
def test_invalidos(secao, chave, valor):
    config = copy.deepcopy(DEFAULTS)
    config[secao][chave] = valor
    with pytest.raises(ValueError):
        validar_config(config)


def test_precedencia_e_consentimento(tmp_path):
    arquivo = tmp_path / 'config.yaml'
    arquivo.write_text(yaml.safe_dump({'roi': {'min_area': .3}, 'security': {'lock_enabled': True},
                                     'benchmark': {'duration': 45}}), encoding='utf-8')
    args, config = interpretar_argumentos(['--config', str(arquivo), '--min-area', '.4', '--duration', '12'])
    assert args.min_area == .4 and config['benchmark']['duration'] == 12
    assert not config['security']['lock_enabled']
    with pytest.raises(SystemExit):
        interpretar_argumentos(['--config', str(arquivo), '--benchmark', '--lock-enabled'])


def test_yaml_invalido(tmp_path):
    arquivo = tmp_path / 'config.yaml'
    arquivo.write_text('roi: [', encoding='utf-8')
    with pytest.raises(ValueError):
        carregar_config(arquivo)


def test_interface_nao_modifica_quadro():
    quadro = np.zeros((480, 640, 3), dtype=np.uint8)
    original = quadro.copy()
    indicadores = dict(app='SentinelCV', versao='1.0.0', modelo='YOLO26N', dispositivo='cpu',
                       fps=10, latencia=20, total=1, validas=1, descartadas=0,
                       confianca=.5, area=.2, limite=10, estado='PRESENTE', restante=10, modo='simulacao')
    tela = renderizar(quadro, [Deteccao(100, 100, 500, 450, .9, 0)], FiltroPresenca(), indicadores)
    assert tela.shape == (617, 950, 3)
    assert np.array_equal(quadro, original)
    assert tela.any()
