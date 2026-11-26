# SentinelCV

Protótipo da disciplina Visão Computacional (PPGTI3007) para detectar pessoas diante de uma estação de trabalho. Nesta etapa, a aplicação apenas apresenta as detecções; não bloqueia a sessão.

## Execução

Com o ambiente virtual existente e as dependências já disponíveis, execute na raiz do projeto:

```bash
.venv/bin/python main.py
```

Pressione **Q** para sair. A câmera é liberada também em caso de erro. O modelo `yolo26n.pt` deve estar disponível localmente; se não estiver, a Ultralytics pode tentar baixá-lo na primeira execução. Os pesos `*.pt` são ignorados pelo Git. Nenhum quadro é salvo ou enviado pela aplicação.

O detector usa o modelo pré-treinado no COCO, classe `person` (0), CPU, tamanho de inferência 320 e confiança mínima 0,50. A inferência recebe o quadro original; as caixas são exibidas em suas coordenadas originais. A latência mede a chamada ao modelo, incluindo seu pré e pós-processamento; o FPS é calculado pela média dos intervalos dos últimos 30 quadros processados. Os parâmetros do detector podem ser alterados no construtor de `Detector`.

## Testes

```bash
.venv/bin/python -m pytest -q
```

Os testes simulam o modelo e não precisam de câmera nem de download de pesos.

## Referência

MAIA, Helton. [Deep Learning — Teoria e Prática, capítulo 4: Visão Computacional](https://heltonmaia.com/deeplearning/chapters/ch4/ch4.html). Exemplos de YOLO26n pré-treinado no COCO, inferência em quadros OpenCV e formatos `xyxy`, `xywh` e `xywhn`.
