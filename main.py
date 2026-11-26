"""Exibe a detecção de pessoas nos quadros da webcam."""

from collections import deque
import sys
import time

import cv2

from src.camera import Camera
from src.detector import Detector


def main() -> int:
    """Executa a visualização até que a tecla Q seja pressionada."""
    camera = Camera()
    try:
        detector = Detector()
        camera.abrir()
        largura, altura, fps, formato = camera.parametros()
        print(f"Parâmetros informados pela câmera: {largura:g} x {altura:g}, "
              f"{fps:g} FPS, FOURCC {formato!r}", flush=True)
        intervalos: deque[float] = deque(maxlen=30)
        anterior = time.perf_counter()
        while True:
            quadro = camera.ler()
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
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
        return 0
    except Exception as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    finally:
        try:
            camera.liberar()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
