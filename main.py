"""Exibe a detecção de pessoas nos quadros da webcam."""

import argparse
from collections import deque
import json
import platform
import sys
import time

import cv2
import torch
import ultralytics

from src.camera import Camera
from src.detector import Detector
from src.metrics import MetricsCollector


def main() -> int:
    """Executa a visualização até que a tecla Q seja pressionada."""
    parser = argparse.ArgumentParser(description="SentinelCV: detecção de pessoas")
    parser.add_argument("--benchmark", action="store_true", help="coleta métricas experimentais")
    parser.add_argument("--duration", type=float, default=60.0, help="duração útil da coleta em segundos")
    parser.add_argument("--scenario", default="cenario", help="identificador do cenário experimental")
    args = parser.parse_args()
    if args.benchmark and (not 0 < args.duration < float("inf")):
        parser.error("--duration deve ser um número positivo e finito")
    camera = Camera()
    metricas = None
    motivo = "erro"
    try:
        detector = Detector()
        camera.abrir()
        largura, altura, fps, formato = camera.parametros()
        print(f"Parâmetros informados pela câmera: {largura:g} x {altura:g}, "
              f"{fps:g} FPS, FOURCC {formato!r}", flush=True)
        if args.benchmark:
            metricas = MetricsCollector(args.duration, args.scenario, {
                "modelo_yolo": str(detector.modelo.model_name),
                "resolucao_inferencia": detector.tamanho_inferencia,
                "resolucao_captura": [largura, altura],
                "confianca_minima": detector.confianca_minima,
                "dispositivo": detector.dispositivo,
                "versoes": {"python": platform.python_version(), "opencv": cv2.__version__,
                            "pytorch": torch.__version__, "ultralytics": ultralytics.__version__},
            })
            print(f"Benchmark: 5 s de aquecimento e {args.duration:g} s de coleta.", flush=True)
        numero_quadro = 0
        intervalos: deque[float] = deque(maxlen=30)
        anterior = time.perf_counter()
        while True:
            inicio_quadro = time.perf_counter()
            quadro = camera.ler()
            numero_quadro += 1
            deteccoes = detector.detectar(quadro)
            for deteccao in deteccoes:
                x1, y1, x2, y2 = map(int, (deteccao.x1, deteccao.y1, deteccao.x2, deteccao.y2))
                cv2.rectangle(quadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(quadro, f"Pessoa {deteccao.confianca:.0%}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            agora = time.perf_counter()
            intervalos.append(agora - anterior)
            anterior = agora
            fps_efetivo = len(intervalos) / sum(intervalos) if sum(intervalos) > 0 else 0.0
            indicadores = [
                (f"FPS: {fps_efetivo:.1f}", (255, 255, 255)),
                (f"Deteccao: {detector.latencia_ms:.1f} ms", (255, 255, 255)),
                (f"Pessoas: {len(deteccoes)}", (255, 255, 255)),
                ("PRESENCA DETECTADA" if deteccoes else "AUSENCIA DETECTADA",
                 (0, 255, 0) if deteccoes else (0, 0, 255)),
            ]
            # Faixa opaca mantém os indicadores legíveis sobre a imagem e as caixas.
            cv2.rectangle(quadro, (0, 0), (min(quadro.shape[1] - 1, 330), 112), (0, 0, 0), -1)
            for indice, (texto, cor) in enumerate(indicadores):
                cv2.putText(quadro, texto, (10, 23 + indice * 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            cv2.imshow("SentinelCV", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            fim_quadro = time.perf_counter()
            concluido = metricas.registrar(
                numero_quadro, inicio_quadro, fim_quadro, detector.latencia_ms,
                [d.confianca for d in deteccoes],
            ) if metricas else False
            if tecla in (ord("q"), ord("Q")):
                motivo = "interrompido_pelo_usuario"
                break
            if concluido:
                motivo = "duracao_concluida"
                break
        return 0
    except Exception as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    finally:
        try:
            camera.liberar()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                if metricas is not None:
                    csv_path, json_path = metricas.exportar(motivo=motivo)
                    print("Resumo do benchmark:")
                    print(json.dumps(metricas.resumo(), ensure_ascii=False, indent=2))
                    print(f"Motivo: {motivo}\nAmostras: {csv_path}\nResumo: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
