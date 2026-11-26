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

## Filtragem geométrica

Por padrão, uma pessoa é considerada válida quando sua caixa ocupa pelo menos 20% do quadro e seu centro está dentro da ROI normalizada `(0.15, 0.20, 0.85, 1.00)`, incluindo as bordas. A ROI é desenhada em azul, as caixas válidas em verde e as descartadas em cinza. A área é uma proporção da imagem, não uma estimativa de distância física.

```bash
.venv/bin/python main.py --min-area 0.20 --roi 0.15 0.20 0.85 1.00
.venv/bin/python main.py --no-roi
.venv/bin/python main.py --benchmark --duration 30 --scenario filtrado
.venv/bin/python main.py --benchmark --duration 30 --scenario baseline --no-roi
```

`--no-roi` desativa os dois critérios geométricos; o detector continua restrito à classe pessoa. O JSON registra os parâmetros e as taxas de presença bruta e válida. No CSV, `detalhes_deteccoes` é uma lista JSON, na ordem das detecções, com coordenadas normalizadas `xywhn` (centro, largura, altura), área relativa, confiança e lista de motivos de descarte. As estatísticas anteriores de presença e confiança continuam se referindo às detecções brutas. Uma lista vazia de motivos indica detecção válida. Experimentos anteriores não são alterados.

## Controle de presença simulado

O temporizador começa na primeira inferência sem presença válida, inclusive durante o aquecimento do benchmark. Após 10 segundos contínuos (configuráveis com `--absence-time`), a aplicação registra `BLOQUEIO SIMULADO` e permanece nesse estado até a tecla **R** reinicializar o controlador. A tecla **Q** encerra a aplicação. Uma detecção descartada não reinicia o temporizador, e falhas são tratadas como erros. Nenhum bloqueio real é executado.

```bash
.venv/bin/python main.py --absence-time 10
.venv/bin/python main.py --benchmark --duration 30 --scenario teste-presenca --absence-time 10
```

Nos novos CSVs, `solicitacao_bloqueio` indica o quadro do evento e `instante_solicitacao_s` registra o instante em segundos desde o início do benchmark, inclusive durante o aquecimento. O instante permanece disponível nos quadros posteriores até a reinicialização. O tempo de ausência para de crescer na solicitação. O benchmark continua após o bloqueio simulado.
