"""Coleta e exportação de métricas agregadas, sem imagens."""

import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import time
from uuid import uuid4

import numpy as np


class MetricsCollector:
    """Registra quadros completos após o aquecimento e resume o experimento."""

    def __init__(self, duracao: float, cenario: str, parametros: dict, aquecimento: float = 5.0) -> None:
        """Configura a duração útil, o cenário e os parâmetros do experimento."""
        if not np.isfinite(duracao) or duracao <= 0:
            raise ValueError("A duração deve ser um número positivo e finito.")
        self.duracao = duracao
        self.cenario = cenario
        self.parametros = parametros
        self.aquecimento = aquecimento
        self.inicio = time.perf_counter()
        self.amostras: list[dict] = []
        self.inicio_coleta: float | None = None
        self.fim_coleta: float | None = None

    def registrar(self, numero: int, inicio: float, fim: float, latencia_ms: float,
                  confiancas: list[float], detalhes: list[dict] | None = None) -> bool:
        """Registra o quadro se iniciado após aquecimento; indica fim da coleta."""
        if inicio - self.inicio < self.aquecimento:
            return False
        if self.inicio_coleta is None:
            self.inicio_coleta = inicio
        tempo_quadro = fim - inicio
        if tempo_quadro <= 0:
            raise ValueError("O tempo do quadro deve ser positivo.")
        self.fim_coleta = fim
        # A API anterior, baseada apenas em confiança, continua utilizável.
        validas = sum(not d["motivos_descarte"] for d in detalhes) if detalhes is not None else len(confiancas)
        areas = [d["area_relativa"] for d in detalhes] if detalhes is not None else []
        self.amostras.append({
            "numero_quadro": numero,
            "tempo_decorrido_s": fim - self.inicio_coleta,
            "latencia_deteccao_ms": latencia_ms,
            "tempo_processamento_s": tempo_quadro,
            "fps_instantaneo": 1 / tempo_quadro,
            "quantidade_pessoas": len(confiancas),
            "presenca": bool(confiancas),
            "maior_confianca": max(confiancas) if confiancas else None,
            "total_deteccoes": len(confiancas),
            "deteccoes_validas": validas,
            "deteccoes_descartadas": len(confiancas) - validas,
            "presenca_bruta": bool(confiancas),
            "presenca_valida": validas > 0,
            "maior_area_relativa": max(areas) if areas else None,
            "detalhes_deteccoes": json.dumps(detalhes if detalhes is not None else [], ensure_ascii=False),
        })
        return fim - self.inicio_coleta >= self.duracao

    def resumo(self) -> dict:
        """Calcula estatísticas; campos sem amostras são representados por null."""
        total = len(self.amostras)
        duracao = (self.fim_coleta - self.inicio_coleta) if total else 0.0
        latencias = [a["latencia_deteccao_ms"] for a in self.amostras]
        fps = [a["fps_instantaneo"] for a in self.amostras]
        confiancas = [a["maior_confianca"] for a in self.amostras if a["maior_confianca"] is not None]
        presentes = sum(a["presenca"] for a in self.amostras)
        validos = sum(a["presenca_valida"] for a in self.amostras)
        return {
            "cenario": self.cenario,
            "parametros": self.parametros,
            "duracao_solicitada_s": self.duracao,
            "aquecimento_s": self.aquecimento,
            "total_quadros": total,
            "duracao_efetiva_s": duracao,
            "fps_medio": total / duracao if duracao else None,
            "fps_instantaneo_minimo": min(fps) if fps else None,
            "fps_instantaneo_maximo": max(fps) if fps else None,
            "latencia_media_ms": statistics.mean(latencias) if latencias else None,
            "latencia_mediana_ms": statistics.median(latencias) if latencias else None,
            "latencia_minima_ms": min(latencias) if latencias else None,
            "latencia_maxima_ms": max(latencias) if latencias else None,
            "latencia_percentil_95_ms": float(np.percentile(latencias, 95)) if latencias else None,
            "latencia_desvio_padrao_ms": statistics.pstdev(latencias) if latencias else None,
            "confianca_media": statistics.mean(confiancas) if confiancas else None,
            "quadros_com_presenca_bruta": presentes,
            "percentual_quadros_com_presenca_bruta": 100 * presentes / total if total else None,
            "quadros_com_presenca_valida": validos,
            "percentual_quadros_com_presenca_valida": 100 * validos / total if total else None,
            "quadros_com_presenca": presentes,
            "quadros_sem_presenca": total - presentes,
            "percentual_quadros_com_presenca": 100 * presentes / total if total else None,
        }

    def exportar(self, diretorio: Path = Path("results"), motivo: str = "concluido") -> tuple[Path, Path]:
        """Salva CSV e JSON com nomes únicos e devolve seus caminhos."""
        diretorio.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.cenario).strip("-_")[:60] or "cenario"
        nome = f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
        csv_path = diretorio / f"{nome}.csv"
        json_path = diretorio / f"{nome}.json"
        campos = ["numero_quadro", "tempo_decorrido_s", "latencia_deteccao_ms",
                  "tempo_processamento_s", "fps_instantaneo", "quantidade_pessoas",
                  "presenca", "maior_confianca", "total_deteccoes", "deteccoes_validas",
                  "deteccoes_descartadas", "presenca_bruta", "presenca_valida",
                  "maior_area_relativa", "detalhes_deteccoes"]
        with csv_path.open("x", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(self.amostras)
        resumo = self.resumo()
        resumo["motivo_encerramento"] = motivo
        with json_path.open("x", encoding="utf-8") as arquivo:
            json.dump(resumo, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
            arquivo.write("\n")
        return csv_path, json_path
