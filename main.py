"""Exibe os quadros da webcam e a taxa de captura observada."""

import sys
import time

import cv2

from src.camera import Camera


def main() -> int:
    """Executa a visualização até que a tecla Q seja pressionada."""
    camera = Camera()
    try:
        camera.abrir()
        largura, altura, fps, formato = camera.parametros()
        print(f"Parâmetros informados pela câmera: {largura:g} x {altura:g}, "
              f"{fps:g} FPS, FOURCC {formato!r}", flush=True)
        anterior = time.perf_counter()
        while True:
            quadro = camera.ler()
            agora = time.perf_counter()
            intervalo = agora - anterior
            anterior = agora
            fps_observado = 1.0 / intervalo if intervalo > 0 else 0.0
            cv2.putText(quadro, f"FPS: {fps_observado:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("SentinelCV", quadro)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
        return 0
    except (RuntimeError, cv2.error) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    finally:
        try:
            camera.liberar()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
