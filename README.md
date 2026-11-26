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

## Experimentos de desempenho

Para coletar métricas, execute, por exemplo:

```bash
.venv/bin/python main.py --benchmark --duration 60 --scenario sala-vazia
.venv/bin/python main.py --benchmark --duration 60 --scenario pessoa-presente
```

A duração é em segundos e vale para a coleta após cinco segundos de aquecimento. O experimento encerra automaticamente após um quadro completo atingir a duração; Q encerra antes e exporta os dados parciais. Cada execução produz um CSV e um JSON com nomes exclusivos em `results/`. Use identificadores de cenário sem dados pessoais. O JSON registra o motivo do encerramento e os parâmetros e versões do ambiente.

O tempo de processamento de cada quadro inclui captura, inferência, desenho, exibição e espera da tecla. O FPS instantâneo é o inverso desse tempo; o FPS médio é o número de amostras dividido pelo tempo decorrido entre o início do primeiro e o fim do último quadro coletado. A duração efetiva pode ultrapassar a solicitada pelo tempo do último quadro. O percentil 95 usa interpolação linear e o desvio padrão é populacional. A confiança média considera a maior confiança de cada quadro com presença; quadros sem detecção têm confiança vazia no CSV. Caso não haja amostras, estatísticas indefinidas aparecem como `null` no JSON.
