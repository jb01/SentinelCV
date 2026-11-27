"""Controle de frequência e captura do quadro mais recente."""
import threading
import time


class Limitador:
    def __init__(self, fps, relogio=time.monotonic, dormir=time.sleep):
        self.periodo = 1 / fps
        self.relogio, self.dormir = relogio, dormir
        self.ultimo = None

    def aguardar(self):
        agora = self.relogio()
        if self.ultimo is not None:
            espera = self.ultimo + self.periodo - agora
            if espera > 0:
                self.dormir(espera)
        self.ultimo = self.relogio()


class CapturaRecente:
    """Descarta quadros anteriores em uma única thread leitora da câmera."""
    def __init__(self, camera):
        self.camera = camera
        self.condicao = threading.Condition()
        self.quadro = None
        self.numero = 0
        self.consumido = 0
        self.erro = None
        self.parar = False
        self.thread = threading.Thread(target=self._capturar, daemon=True, name='captura')
        self.thread.start()

    def _capturar(self):
        try:
            while not self.parar:
                quadro = self.camera.ler()
                with self.condicao:
                    self.quadro = quadro
                    self.numero += 1
                    self.condicao.notify_all()
        except Exception as erro:
            with self.condicao:
                self.erro = erro
                self.condicao.notify_all()

    def ler(self):
        with self.condicao:
            while self.numero == self.consumido and self.erro is None:
                self.condicao.wait(timeout=0.5)
            if self.erro is not None:
                raise RuntimeError('Falha na captura contínua') from self.erro
            self.consumido = self.numero
            return self.quadro

    def liberar(self):
        self.parar = True
        self.thread.join(timeout=2)
        self.camera.liberar()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('A thread de captura não encerrou.')
