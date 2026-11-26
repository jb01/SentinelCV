"""Testes da coleta e das estatísticas sem câmera ou modelo."""

import csv
import json
from unittest.mock import patch

import pytest

from src.metrics import MetricsCollector


@pytest.fixture
def coletor() -> MetricsCollector:
    """Inicializa o relógio em um instante conhecido."""
    with patch("src.metrics.time.perf_counter", return_value=100.0):
        return MetricsCollector(3, "sala", {"modelo_yolo": "yolo26n.pt"})


def test_aquecimento_e_estatisticas(coletor: MetricsCollector) -> None:
    assert not coletor.registrar(1, 104.9, 105.2, 999, [0.1])
    assert not coletor.registrar(2, 105, 106, 10, [0.8, 0.9])
    assert not coletor.registrar(3, 106, 107, 20, [])
    assert coletor.registrar(4, 107, 108, 30, [0.7])
    resumo = coletor.resumo()
    assert resumo["total_quadros"] == 3
    assert resumo["duracao_efetiva_s"] == 3
    assert resumo["fps_medio"] == 1
    assert resumo["fps_instantaneo_minimo"] == resumo["fps_instantaneo_maximo"] == 1
    assert resumo["latencia_media_ms"] == resumo["latencia_mediana_ms"] == 20
    assert resumo["latencia_minima_ms"] == 10
    assert resumo["latencia_maxima_ms"] == 30
    assert resumo["latencia_percentil_95_ms"] == pytest.approx(29)
    assert resumo["latencia_desvio_padrao_ms"] == pytest.approx((200 / 3) ** 0.5)
    assert resumo["confianca_media"] == pytest.approx(0.8)
    assert resumo["quadros_com_presenca"] == 2
    assert resumo["quadros_sem_presenca"] == 1
    assert resumo["percentual_quadros_com_presenca"] == pytest.approx(200 / 3)
    assert coletor.amostras[1]["maior_confianca"] is None


def test_sem_amostras_e_sem_presenca(coletor: MetricsCollector) -> None:
    assert coletor.resumo()["fps_medio"] is None
    assert coletor.resumo()["latencia_media_ms"] is None
    coletor.registrar(1, 105, 105.5, 5, [])
    resumo = coletor.resumo()
    assert resumo["confianca_media"] is None
    assert resumo["fps_medio"] == 2
    assert resumo["latencia_desvio_padrao_ms"] == 0
    assert resumo["percentual_quadros_com_presenca"] == 0


def test_exportacao_unica_e_confianca_ausente(coletor: MetricsCollector, tmp_path) -> None:
    coletor.registrar(10, 105, 106, 15, [])
    csv_path, json_path = coletor.exportar(tmp_path, "interrompido_pelo_usuario")
    outro_csv, outro_json = coletor.exportar(tmp_path)
    assert csv_path != outro_csv and json_path != outro_json
    with csv_path.open(encoding="utf-8", newline="") as arquivo:
        linha = next(csv.DictReader(arquivo))
    assert linha["numero_quadro"] == "10"
    assert linha["maior_confianca"] == ""
    resumo = json.loads(json_path.read_text(encoding="utf-8"))
    assert resumo["motivo_encerramento"] == "interrompido_pelo_usuario"
    assert resumo["confianca_media"] is None
    assert resumo["parametros"]["modelo_yolo"] == "yolo26n.pt"


@pytest.mark.parametrize("duracao", [0, -1, float("nan"), float("inf")])
def test_duracao_invalida(duracao: float) -> None:
    with pytest.raises(ValueError):
        MetricsCollector(duracao, "teste", {})


def test_exportacao_filtragem(coletor: MetricsCollector, tmp_path) -> None:
    """Preserva os dados individuais e distingue presença bruta da válida."""
    detalhes = [{"coordenadas_xywhn": (0.1, 0.1, 0.1, 0.1), "area_relativa": 0.01,
                 "confianca": 0.8, "motivos_descarte": ["area_insuficiente", "centro_fora_roi"]}]
    coletor.registrar(1, 105, 106, 10, [0.8], detalhes)
    coletor.registrar(2, 106, 107, 10, [], [])
    csv_path, json_path = coletor.exportar(tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert linhas[0]["deteccoes_descartadas"] == "1"
    assert linhas[0]["presenca_bruta"] == "True"
    assert linhas[0]["presenca_valida"] == "False"
    assert json.loads(linhas[0]["detalhes_deteccoes"])[0]["motivos_descarte"] == detalhes[0]["motivos_descarte"]
    assert linhas[1]["maior_area_relativa"] == ""
    resumo = json.loads(json_path.read_text(encoding="utf-8"))
    assert resumo["quadros_com_presenca_bruta"] == 1
    assert resumo["percentual_quadros_com_presenca_bruta"] == 50
    assert resumo["quadros_com_presenca_valida"] == 0
    assert resumo["percentual_quadros_com_presenca_valida"] == 0
