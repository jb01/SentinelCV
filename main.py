"""Exibe a detecção de pessoas nos quadros da webcam."""

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import platform
import sys
import time

import cv2
import torch
import ultralytics

from src.camera import Camera
from src.detector import Detector
from src.metrics import MetricsCollector
from src.presence import ControladorPresenca, EstadoPresenca
from src.roi import FiltroPresenca, ROI
from src.screen_lock import BloqueadorSessao


def configurar_log() -> logging.Logger:
    """Configura registro local de eventos sem informações pessoais."""
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("sentinelcv")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler("logs/sentinelcv.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def registrar_evento(logger: logging.Logger, tipo: str, anterior: EstadoPresenca,
                     controlador: ControladorPresenca, modo: str, resultado=None,
                     ausencia: float | None = None) -> None:
    """Registra somente dados operacionais estruturados."""
    logger.info(json.dumps({"evento": tipo, "estado_anterior": anterior.name,
                            "novo_estado": controlador.estado.name,
                            "tempo_ausencia_s": controlador.tempo_ausencia if ausencia is None else ausencia,
                            "limite_s": controlador.tempo_limite, "modo": modo,
                            "resultado": resultado.mensagem if resultado else None,
                            "latencia_ms": resultado.latencia_ms if resultado else None}, ensure_ascii=False))


def main() -> int:
    """Executa a visualização até que a tecla Q seja pressionada."""
    parser = argparse.ArgumentParser(description="SentinelCV: detecção de pessoas")
    parser.add_argument("--benchmark", action="store_true", help="coleta métricas experimentais")
    parser.add_argument("--duration", type=float, default=60.0, help="duração útil da coleta em segundos")
    parser.add_argument("--scenario", default="cenario", help="identificador do cenário experimental")
    parser.add_argument("--min-area", type=float, default=0.20, help="área relativa mínima (0 a 1)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        default=(0.15, 0.20, 0.85, 1.0), help="limites normalizados da ROI")
    parser.add_argument("--no-roi", action="store_true", help="desabilita a filtragem geométrica")
    parser.add_argument("--absence-time", type=float, default=10.0, help="limite de ausência em segundos")
    parser.add_argument("--lock-enabled", action="store_true", help="ativa solicitação real de bloqueio")
    args = parser.parse_args()
    if args.benchmark and args.lock_enabled:
        parser.error("--benchmark é incompatível com --lock-enabled")
    try:
        filtro = FiltroPresenca(ROI(*args.roi), args.min_area, not args.no_roi)
        controlador = ControladorPresenca(args.absence_time, relogio=time.perf_counter)
    except ValueError as erro:
        parser.error(str(erro))
    if args.benchmark and (not 0 < args.duration < float("inf")):
        parser.error("--duration deve ser um número positivo e finito")
    camera = Camera()
    bloqueador = BloqueadorSessao(modo_simulado=not args.lock_enabled)
    modo = "real" if args.lock_enabled else "simulacao"
    logger = configurar_log()
    registrar_evento(logger, "inicio", controlador.estado, controlador, modo)
    etapa = "inicializacao"
    acao_executada = False
    metricas = None
    motivo = "erro"
    try:
        etapa = "inferencia"
        detector = Detector()
        etapa = "captura"
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
                "roi": list(args.roi),
                "area_relativa_minima": filtro.min_area,
                "filtragem_geometrica_ativa": filtro.habilitado,
                "tempo_limite_ausencia_s": controlador.tempo_limite,
                "versoes": {"python": platform.python_version(), "opencv": cv2.__version__,
                            "pytorch": torch.__version__, "ultralytics": ultralytics.__version__},
            })
            print(f"Benchmark: 5 s de aquecimento e {args.duration:g} s de coleta.", flush=True)
        numero_quadro = 0
        intervalos: deque[float] = deque(maxlen=30)
        anterior = time.perf_counter()
        while True:
            inicio_quadro = time.perf_counter()
            etapa = "captura"
            quadro = camera.ler()
            numero_quadro += 1
            etapa = "inferencia"
            deteccoes = detector.detectar(quadro)
            etapa = "processamento"
            altura_quadro, largura_quadro = quadro.shape[:2]
            validas, descartadas = filtro.classificar(deteccoes, largura_quadro, altura_quadro)
            anterior_estado = controlador.estado
            controlador.atualizar(bool(validas))
            if anterior_estado != controlador.estado:
                registrar_evento(logger, "transicao_presenca", anterior_estado, controlador, modo)
            solicitacao = controlador.solicitar_bloqueio
            if solicitacao:
                acao_executada = True
                resultado = bloqueador.bloquear()
                anterior_estado = controlador.estado
                if not resultado.sucesso_solicitacao:
                    ausencia = controlador.tempo_ausencia
                    controlador.registrar_erro("falha na solicitação de bloqueio")
                    registrar_evento(logger, "falha_bloqueio", anterior_estado, controlador, modo, resultado, ausencia)
                    print(f"Erro de bloqueio: {resultado.mensagem}", file=sys.stderr)
                    return 1
                if resultado.modo_simulado:
                    controlador.confirmar_bloqueio()
                registrar_evento(logger, "solicitacao_bloqueio", anterior_estado, controlador, modo, resultado)
                print(resultado.mensagem, flush=True)
                if not resultado.modo_simulado:
                    return 0
            tempo_ausencia = controlador.tempo_ausencia
            tempo_restante = controlador.tempo_restante
            detalhes = []
            if filtro.habilitado:
                roi = filtro.roi
                cv2.rectangle(quadro, (int(roi.xmin * largura_quadro), int(roi.ymin * altura_quadro)),
                              (min(largura_quadro - 1, int(roi.xmax * largura_quadro)),
                               min(altura_quadro - 1, int(roi.ymax * altura_quadro))), (255, 0, 0), 2)
            for deteccao in deteccoes:
                motivos = filtro.motivos_descarte(deteccao, largura_quadro, altura_quadro)
                area = deteccao.area_relativa(largura_quadro, altura_quadro)
                detalhes.append({"coordenadas_xywhn": deteccao.coordenadas_normalizadas(largura_quadro, altura_quadro),
                                 "area_relativa": area, "confianca": deteccao.confianca,
                                 "motivos_descarte": motivos})
                cor = (128, 128, 128) if motivos else (0, 255, 0)
                x1, y1, x2, y2 = map(int, (deteccao.x1, deteccao.y1, deteccao.x2, deteccao.y2))
                cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, 2)
                cv2.circle(quadro, tuple(map(int, deteccao.centro)), 4, cor, -1)
                cv2.putText(quadro, f"Pessoa {deteccao.confianca:.0%} Area {area:.1%}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            agora = time.perf_counter()
            intervalos.append(agora - anterior)
            anterior = agora
            fps_efetivo = len(intervalos) / sum(intervalos) if sum(intervalos) > 0 else 0.0
            cor_estado = ((0, 255, 0) if controlador.estado == EstadoPresenca.PRESENTE else
                          (0, 255, 255) if controlador.estado == EstadoPresenca.AUSENCIA_TEMPORARIA else
                          (0, 0, 255))
            indicadores = [
                ("BLOQUEIO REAL ATIVO" if args.lock_enabled else "MODO SIMULACAO", (255, 255, 255)),
                (f"FPS: {fps_efetivo:.1f}  Deteccao: {detector.latencia_ms:.1f} ms", (255, 255, 255)),
                (f"Pessoas: {len(deteccoes)}  Validas: {len(validas)}  Descartadas: {len(descartadas)}", (255, 255, 255)),
                (f"Presenca bruta: {'SIM' if deteccoes else 'NAO'}  Valida: {'SIM' if validas else 'NAO'}", (255, 255, 255)),
                (f"Estado: {controlador.estado.name}", cor_estado),
                (f"Ausencia: {tempo_ausencia:.1f}s  Restante: {tempo_restante:.1f}s", cor_estado),
                ("BLOQUEIO SIMULADO - R reinicia" if controlador.estado == EstadoPresenca.BLOQUEADO
                 else f"Contagem regressiva: {tempo_restante:.1f}s", cor_estado),
            ]
            cv2.rectangle(quadro, (0, 0), (min(largura_quadro - 1, 500), 198), (0, 0, 0), -1)
            for indice, (texto, cor) in enumerate(indicadores):
                cv2.putText(quadro, texto, (10, 21 + indice * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, cor, 1)
            barra_largura = min(300, max(0, largura_quadro - 22))
            progresso = min(1.0, tempo_ausencia / controlador.tempo_limite)
            cv2.rectangle(quadro, (10, 174), (10 + barra_largura, 187), (220, 220, 220), 1)
            if progresso > 0:
                cv2.rectangle(quadro, (11, 175), (10 + int(barra_largura * progresso), 186), cor_estado, -1)
            cv2.imshow("SentinelCV", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            fim_quadro = time.perf_counter()
            concluido = metricas.registrar(
                numero_quadro, inicio_quadro, fim_quadro, detector.latencia_ms,
                [d.confianca for d in deteccoes], detalhes,
                estado_presenca=controlador.estado.name, tempo_ausencia_s=tempo_ausencia,
                solicitacao_bloqueio=solicitacao,
                instante_solicitacao_s=(controlador.instante_solicitacao - metricas.inicio
                                         if controlador.instante_solicitacao is not None else None),
            ) if metricas else False
            if tecla in (ord("r"), ord("R")):
                anterior_estado = controlador.estado
                controlador.reinicializar()
                bloqueador.reinicializar()
                acao_executada = False
                registrar_evento(logger, "reinicializacao", anterior_estado, controlador, modo)
                print("Controlador de presença reinicializado.", flush=True)
            if tecla in (ord("q"), ord("Q")):
                motivo = "interrompido_pelo_usuario"
                break
            if concluido:
                motivo = "duracao_concluida"
                break
        return 0
    except Exception as erro:
        anterior_estado = controlador.estado
        ausencia = controlador.tempo_ausencia
        controlador.registrar_erro(f"falha de {etapa}")
        registrar_evento(logger, f"falha_{etapa}", anterior_estado, controlador, modo, ausencia=ausencia)
        print(f"Erro de {etapa}: {erro}", file=sys.stderr)
        if etapa in ("captura", "inferencia") and not acao_executada:
            resultado = bloqueador.bloquear()
            registrar_evento(logger, "bloqueio_preventivo", controlador.estado, controlador, modo, resultado, ausencia)
            print(f"Bloqueio preventivo: {resultado.mensagem}", file=sys.stderr)
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
