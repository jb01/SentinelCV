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

from src.config import carregar_config, validar_config
from src.interface import renderizar
from src.camera import Camera
from src.detector import Detector
from src.metrics import MetricsCollector
from src.presence import ControladorPresenca, EstadoPresenca
from src.roi import FiltroPresenca, ROI
from src.screen_lock import BloqueadorSessao


def configurar_log(arquivo="logs/sentinelcv.log", habilitado=True) -> logging.Logger:
    """Configura registro local de eventos sem informações pessoais."""
    if habilitado:
        Path(arquivo).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sentinelcv")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    if habilitado:
        handler = logging.FileHandler(arquivo, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
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


def interpretar_argumentos(argv=None):
    """Combina configuração validada e opções explícitas da CLI."""
    parser = argparse.ArgumentParser(description="SentinelCV: detecção de pessoas")
    parser.add_argument("--config", default=Path(__file__).resolve().parent / "config.yaml", help="arquivo YAML de configuração")
    parser.add_argument("--benchmark", action="store_true", help="coleta métricas experimentais")
    parser.add_argument("--duration", type=float, default=None, help="duração útil da coleta em segundos")
    parser.add_argument("--scenario", default="cenario", help="identificador do cenário experimental")
    parser.add_argument("--min-area", type=float, default=None, help="área relativa mínima (0 a 1)")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        default=None, help="limites normalizados da ROI")
    parser.add_argument("--no-roi", action="store_true", help="desabilita a filtragem geométrica")
    parser.add_argument("--absence-time", type=float, default=None, help="limite de ausência em segundos")
    parser.add_argument("--lock-enabled", action="store_true", help="ativa solicitação real de bloqueio")
    args = parser.parse_args(argv)
    try:
        config = carregar_config(args.config)
        if args.duration is not None:
            config["benchmark"]["duration"] = args.duration
        if args.min_area is not None:
            config["roi"]["min_area"] = args.min_area
        if args.roi is not None:
            for chave, valor in zip(("xmin", "ymin", "xmax", "ymax"), args.roi):
                config["roi"][chave] = valor
        if args.no_roi:
            config["roi"]["enabled"] = False
        if args.absence_time is not None:
            config["presence"]["absence_timeout"] = args.absence_time
        config["security"]["lock_enabled"] = args.lock_enabled
        validar_config(config)
    except ValueError as erro:
        parser.error(str(erro))
    args.duration = config["benchmark"]["duration"]
    args.min_area = config["roi"]["min_area"]
    args.roi = [config["roi"][k] for k in ("xmin", "ymin", "xmax", "ymax")]
    args.no_roi = not config["roi"]["enabled"]
    args.absence_time = config["presence"]["absence_timeout"]
    if args.benchmark and args.lock_enabled:
        parser.error("--benchmark é incompatível com --lock-enabled")
    try:
        filtro = FiltroPresenca(ROI(*args.roi), args.min_area, not args.no_roi)
        controlador = ControladorPresenca(args.absence_time, relogio=time.perf_counter)
    except ValueError as erro:
        parser.error(str(erro))
    if args.benchmark and (not 0 < args.duration < float("inf")):
        parser.error("--duration deve ser um número positivo e finito")
    return args, config


def main() -> int:
    """Executa a visualização até que a tecla Q seja pressionada."""
    args, config = interpretar_argumentos()
    filtro = FiltroPresenca(ROI(*args.roi), args.min_area, not args.no_roi)
    controlador = ControladorPresenca(args.absence_time, relogio=time.perf_counter)
    c = config["camera"]
    camera = Camera(c["index"], c["width"], c["height"], c["fps"], c["codec"])
    bloqueador = BloqueadorSessao(modo_simulado=not args.lock_enabled)
    modo = "benchmark" if args.benchmark else "real" if args.lock_enabled else "simulacao"
    logger = configurar_log(config["logging"]["file"], config["logging"]["enabled"])
    registrar_evento(logger, "inicio", controlador.estado, controlador, modo)
    etapa = "inicializacao"
    acao_executada = False
    metricas = None
    motivo = "erro"
    try:
        etapa = "inferencia"
        d = config["detector"]
        detector = Detector(d["model"], d["device"], d["imgsz"], d["confidence"], d["classes"])
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
                "configuracao_efetiva": config,
                "versoes": {"python": platform.python_version(), "opencv": cv2.__version__,
                            "pytorch": torch.__version__, "ultralytics": ultralytics.__version__},
            }, aquecimento=config["benchmark"]["warmup"])
            print(f"Benchmark: {config['benchmark']['warmup']:g} s de aquecimento e {args.duration:g} s de coleta.", flush=True)
        ajuda = True
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
            for deteccao in deteccoes:
                motivos = filtro.motivos_descarte(deteccao, largura_quadro, altura_quadro)
                detalhes.append({"coordenadas_xywhn": deteccao.coordenadas_normalizadas(largura_quadro, altura_quadro),
                                 "area_relativa": deteccao.area_relativa(largura_quadro, altura_quadro),
                                 "confianca": deteccao.confianca, "motivos_descarte": motivos})
            agora = time.perf_counter()
            intervalos.append(agora - anterior)
            anterior = agora
            fps_efetivo = len(intervalos) / sum(intervalos) if sum(intervalos) > 0 else 0.0
            interface = renderizar(quadro, deteccoes, filtro, {
                "app": config["app"]["name"], "versao": config["app"]["version"],
                "modelo": Path(d["model"]).stem.upper(), "dispositivo": d["device"],
                "fps": fps_efetivo, "latencia": detector.latencia_ms, "total": len(deteccoes),
                "validas": len(validas), "descartadas": len(descartadas),
                "confianca": d["confidence"], "area": filtro.min_area,
                "limite": controlador.tempo_limite, "estado": controlador.estado.name,
                "restante": tempo_restante, "modo": modo,
            }, ajuda)
            cv2.imshow("SentinelCV", interface)
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
            if tecla in (ord("h"), ord("H")):
                ajuda = not ajuda
            if tecla in (ord("r"), ord("R")) and not args.lock_enabled:
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
        if config["security"]["fail_safe"] and etapa in ("captura", "inferencia") and not acao_executada:
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
                    csv_path, json_path = metricas.exportar(diretorio=Path(config["benchmark"]["output_dir"]), motivo=motivo)
                    print("Resumo do benchmark:")
                    print(json.dumps(metricas.resumo(), ensure_ascii=False, indent=2))
                    print(f"Motivo: {motivo}\nAmostras: {csv_path}\nResumo: {json_path}")


if __name__ == "__main__":
    sys.exit(main())
