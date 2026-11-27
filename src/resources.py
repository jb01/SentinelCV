"""Amostras periódicas de recursos do processo, sem imagens."""
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import threading
import time
from uuid import uuid4

import numpy as np


class Recursos:
    def __init__(self, intervalo, config, psutil_module=None):
        if psutil_module is None:
            try:
                import psutil as psutil_module
            except ImportError as erro:
                raise RuntimeError('Monitoramento de recursos requer psutil; instale-o no ambiente.') from erro
        self.processo = psutil_module.Process()
        self.cpus = psutil_module.cpu_count(logical=True) or 1
        self.intervalo, self.config = intervalo, config
        self.amostras = []
        self.inicio = time.monotonic()
        self.lock = threading.Lock()
        self.fps = self.latencia = None
        self.parar = threading.Event()
        self.processo.cpu_percent(None)
        self.thread = threading.Thread(target=self._executar, daemon=True, name='recursos')

    def iniciar(self):
        self.thread.start()

    def atualizar(self, fps, latencia):
        with self.lock:
            self.fps, self.latencia = fps, latencia

    def coletar(self):
        with self.lock:
            fps, latencia = self.fps, self.latencia
        cpu = self.processo.cpu_percent(None)
        self.amostras.append({'tempo_decorrido_s': time.monotonic() - self.inicio,
                              'cpu_processo_pct': cpu, 'cpu_normalizada_pct': cpu / self.cpus,
                              'rss_mib': self.processo.memory_info().rss / (1024 ** 2),
                              'ram_pct': self.processo.memory_percent(),
                              'threads': self.processo.num_threads(), 'fps_efetivo': fps,
                              'latencia_inferencia_ms': latencia})

    def _executar(self):
        while not self.parar.wait(self.intervalo):
            self.coletar()

    def finalizar(self):
        self.parar.set()
        self.thread.join()

    def exportar(self, diretorio, motivo):
        diretorio = Path(diretorio)
        diretorio.mkdir(parents=True, exist_ok=True)
        nome = f"recursos_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
        campos = ['tempo_decorrido_s', 'cpu_processo_pct', 'cpu_normalizada_pct', 'rss_mib', 'ram_pct', 'threads', 'fps_efetivo', 'latencia_inferencia_ms']
        csv_path, json_path = diretorio / (nome + '.csv'), diretorio / (nome + '.json')
        with csv_path.open('x', encoding='utf-8', newline='') as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(self.amostras)
        def valores(campo):
            return [a[campo] for a in self.amostras if a[campo] is not None]
        def media(campo):
            dados = valores(campo)
            return statistics.mean(dados) if dados else None
        cpu, ram = valores('cpu_processo_pct'), valores('rss_mib')
        resumo = {'configuracao_efetiva': self.config, 'motivo_encerramento': motivo,
                  'cpus_logicas': self.cpus, 'total_amostras': len(self.amostras),
                  'cpu_processo_media_pct': media('cpu_processo_pct'),
                  'cpu_processo_maxima_pct': max(cpu) if cpu else None,
                  'cpu_processo_p95_pct': float(np.percentile(cpu, 95)) if cpu else None,
                  'cpu_normalizada_media_pct': media('cpu_normalizada_pct'),
                  'ram_media_mib': media('rss_mib'), 'ram_maxima_mib': max(ram) if ram else None,
                  'variacao_ram_mib': ram[-1] - ram[0] if len(ram) >= 2 else None,
                  'fps_medio': media('fps_efetivo'), 'latencia_media_ms': media('latencia_inferencia_ms'),
                  'amostras': self.amostras}
        with json_path.open('x', encoding='utf-8') as arquivo:
            json.dump(resumo, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
        return csv_path, json_path
