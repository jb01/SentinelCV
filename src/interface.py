"""Composição da interface OpenCV após a inferência."""
import cv2
import numpy as np


def renderizar(quadro, deteccoes, filtro, indicadores, ajuda=True):
    """Devolve uma nova imagem, preservando o quadro usado na inferência."""
    video = quadro.copy()
    altura, largura = video.shape[:2]
    if filtro.habilitado:
        r = filtro.roi
        cv2.rectangle(video, (int(r.xmin * largura), int(r.ymin * altura)),
                      (min(largura - 1, int(r.xmax * largura)), min(altura - 1, int(r.ymax * altura))), (230, 150, 45), 2)
    for d in deteccoes:
        cor = (135, 135, 135) if filtro.motivos_descarte(d, largura, altura) else (80, 210, 100)
        x1, y1, x2, y2 = map(int, (d.x1, d.y1, d.x2, d.y2))
        cv2.rectangle(video, (x1, y1), (x2, y2), cor, 2)
        cv2.circle(video, tuple(map(int, d.centro)), 4, cor, -1)
        cv2.putText(video, f'{d.confianca:.0%} {d.area_relativa(largura, altura):.1%}', (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, cor, 1, cv2.LINE_AA)
    painel = 310
    cabecalho, rodape = 72, 65
    tela = np.full ( (altura + cabecalho + rodape, largura + painel, 3), (24, 23, 21), dtype=np.uint8)
    tela[cabecalho:cabecalho + altura, :largura] = video
    cv2.rectangle(tela, (largura, cabecalho), (largura + painel, cabecalho + altura), (37, 35, 32), -1)
    def texto(s, x, y, cor=(220, 220, 220), escala=.48):
        cv2.putText(tela, str(s), (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)
    texto(f"{indicadores['app']} v{indicadores['versao']}", 16, 29, (245, 245, 245), .7)
    texto('Computer Vision for Information Security', 16, 52, (160, 165, 170), .45)
    estado = indicadores['estado']
    cor = ((80, 210, 100) if estado == 'PRESENTE' else (40, 210, 235) if estado in ('INICIALIZANDO', 'AUSENCIA_TEMPORARIA') else (30, 145, 255) if estado == 'ERRO' else (70, 70, 240))
    linhas = [f"Modelo: {indicadores['modelo']}", f"Dispositivo: {indicadores['dispositivo'].upper()}",
              f"FPS: {indicadores['fps']:.1f}", f"Latencia: {indicadores['latencia']:.1f} ms",
              f"Pessoas: {indicadores['total']}", f"Validas: {indicadores['validas']}",
              f"Descartadas: {indicadores['descartadas']}", f"Confianca minima: {indicadores['confianca']:.0%}",
              f"Area minima: {indicadores['area']:.0%}", f"Limite ausencia: {indicadores['limite']:.1f} s",
              f"Estado: {estado}", f"Restante: {indicadores['restante']:.1f} s", f"Modo: {indicadores['modo'].upper()}"]
    espacamento = min(28, max(12, (altura - 35) // len(linhas)))
    for i, linha in enumerate(linhas):
        texto(linha, largura + 14, cabecalho + 24 + i * espacamento, cor if i == 10 else (220, 220, 220), .43)
    y = cabecalho + altura + 20
    texto(f"STATUS: {estado}  |  Restante: {indicadores['restante']:.1f}s", 15, y, cor)
    if ajuda:
        texto('Q: sair   R: reiniciar simulacao   H: ocultar ajuda', 15, y + 24, (170, 175, 180), .43)
    else:
        texto('H: ajuda', 15, y + 24, (170, 175, 180), .43)
    x = largura + 14
    cv2.rectangle(tela, (x, y - 10), (largura + painel - 15, y + 2), (110, 110, 110), 1)
    progresso = max(0., min(1., 1 - indicadores['restante'] / indicadores['limite']))
    if progresso:
        cv2.rectangle(tela, (x + 1, y - 9), (x + int((painel - 30) * progresso), y + 1), cor, -1)
    return tela
