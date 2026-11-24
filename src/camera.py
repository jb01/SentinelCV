"""Captura de quadros de uma webcam usando o backend V4L2."""

import cv2
import numpy as np
from numpy.typing import NDArray


class Camera:
    """Gerencia a abertura, a leitura e a liberação de uma câmera."""

    def __init__(
        self, indice: int = 0, largura: int = 640, altura: int = 480,
        fps: float = 30.0, formato: str = "MJPG",
    ) -> None:
        """Define o dispositivo e os parâmetros solicitados à câmera."""
        if len(formato) != 4:
            raise ValueError("O formato deve ser um código FOURCC de quatro caracteres.")
        self.indice = indice
        self.largura = largura
        self.altura = altura
        self.fps = fps
        self.formato = formato
        self._captura: cv2.VideoCapture | None = None

    def abrir(self) -> None:
        """Abre a câmera e solicita os parâmetros configurados.

        Levanta RuntimeError se o dispositivo não puder ser aberto.
        """
        if self._captura is not None:
            raise RuntimeError("A câmera já está aberta.")
        captura = cv2.VideoCapture(self.indice, cv2.CAP_V4L2)
        self._captura = captura
        if not captura.isOpened():
            self.liberar()
            raise RuntimeError(f"Não foi possível abrir a câmera de índice {self.indice}.")
        captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.formato))
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.largura)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.altura)
        captura.set(cv2.CAP_PROP_FPS, self.fps)

    def parametros(self) -> tuple[float, float, float, str]:
        """Retorna largura, altura, FPS e FOURCC informados pelo dispositivo."""
        if self._captura is None or not self._captura.isOpened():
            raise RuntimeError("A câmera não está aberta.")
        captura = self._captura
        codigo = int(captura.get(cv2.CAP_PROP_FOURCC))
        formato = "".join(chr((codigo >> (8 * i)) & 0xFF) for i in range(4))
        return (captura.get(cv2.CAP_PROP_FRAME_WIDTH),
                captura.get(cv2.CAP_PROP_FRAME_HEIGHT),
                captura.get(cv2.CAP_PROP_FPS), formato)

    def ler(self) -> NDArray[np.uint8]:
        """Retorna um quadro ou levanta RuntimeError em caso de falha."""
        if self._captura is None or not self._captura.isOpened():
            raise RuntimeError("A câmera não está aberta.")
        sucesso, quadro = self._captura.read()
        if not sucesso or quadro is None:
            raise RuntimeError("Não foi possível ler um quadro da câmera.")
        return quadro

    def liberar(self) -> None:
        """Libera o dispositivo, caso esteja aberto."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
