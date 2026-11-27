"""Carregamento e validação da configuração operacional."""
from copy import deepcopy
import math
from pathlib import Path

import yaml

DEFAULTS = {
    'app': {'name': 'SentinelCV', 'version': '1.0.0'},
    'camera': {'index': 0, 'width': 640, 'height': 480, 'fps': 30, 'codec': 'MJPG'},
    'detector': {'model': 'yolo26n.pt', 'device': 'cpu', 'imgsz': 320, 'confidence': 0.5, 'classes': [0]},
    'roi': {'enabled': True, 'xmin': 0.15, 'ymin': 0.20, 'xmax': 0.85, 'ymax': 1.0, 'min_area': 0.20},
    'presence': {'absence_timeout': 10.0},
    'security': {'lock_enabled': False, 'fail_safe': True},
    'benchmark': {'duration': 30, 'warmup': 5, 'output_dir': 'results'},
    'logging': {'enabled': True, 'file': 'logs/sentinelcv.log'},
}


def validar_config(config):
    """Rejeita valores inválidos antes de acessar câmera ou modelo."""
    def numero(valor, minimo, maximo=None, inclusivo=True):
        return (isinstance(valor, (int, float)) and not isinstance(valor, bool)
                and math.isfinite(valor) and (valor >= minimo if inclusivo else valor > minimo)
                and (maximo is None or valor <= maximo))
    def inteiro(valor, minimo):
        return isinstance(valor, int) and not isinstance(valor, bool) and valor >= minimo
    c, d, r = config['camera'], config['detector'], config['roi']
    if not inteiro(c['index'], 0) or not all(inteiro(c[k], 1) for k in ('width', 'height')) or not numero(c['fps'], 0, inclusivo=False) or not isinstance(c['codec'], str) or len(c['codec']) != 4:
        raise ValueError('Configuração de câmera inválida.')
    if (not isinstance(d['model'], str) or not d['model'].strip() or d['device'] != 'cpu'
            or not inteiro(d['imgsz'], 1) or not numero(d['confidence'], 0, 1)
            or d['classes'] != [0] or any(type(x) is not int for x in d['classes'])):
        raise ValueError('Configuração de detector inválida: somente CPU e classe person são permitidas.')
    if (type(r['enabled']) is not bool or not all(numero(r[k], 0, 1) for k in ('xmin', 'ymin', 'xmax', 'ymax', 'min_area'))
            or r['xmin'] >= r['xmax'] or r['ymin'] >= r['ymax']):
        raise ValueError('Configuração de ROI inválida.')
    if not numero(config['presence']['absence_timeout'], 0, inclusivo=False):
        raise ValueError('Tempo de ausência inválido.')
    b = config['benchmark']
    if not numero(b['duration'], 0, inclusivo=False) or not numero(b['warmup'], 0):
        raise ValueError('Duração ou aquecimento do benchmark inválido.')
    for sec, keys in (('security', ('lock_enabled', 'fail_safe')), ('logging', ('enabled',))):
        if any(type(config[sec][k]) is not bool for k in keys):
            raise ValueError(f'Configuração {sec} inválida.')
    for sec, key in (('app', 'name'), ('app', 'version'), ('benchmark', 'output_dir'), ('logging', 'file')):
        if not isinstance(config[sec][key], str) or not config[sec][key].strip():
            raise ValueError(f'Configuração {sec}.{key} inválida.')
    return config


def carregar_config(caminho='config.yaml'):
    """Carrega YAML seguro, preenchendo campos omitidos com valores padrão."""
    try:
        with Path(caminho).open(encoding='utf-8') as arquivo:
            dados = yaml.safe_load(arquivo)
    except (OSError, yaml.YAMLError) as erro:
        raise ValueError(f'Não foi possível carregar a configuração: {erro}') from erro
    if dados is None:
        dados = {}
    if not isinstance(dados, dict):
        raise ValueError('A configuração deve ser um mapeamento.')
    config = deepcopy(DEFAULTS)
    for secao, valores in dados.items():
        if secao not in config or not isinstance(valores, dict):
            raise ValueError(f'Seção de configuração inválida: {secao}')
        desconhecidas = valores.keys() - config[secao].keys()
        if desconhecidas:
            raise ValueError(f'Parâmetros desconhecidos em {secao}: {desconhecidas}')
        config[secao].update(valores)
    return validar_config(config)
