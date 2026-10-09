# Dataset simulado do kit da Fase 2 (classe `lipo`)

Gerado em 2026-10-09 no BiguaSim (engine com rotação no `SpawnMesh`; o kit
que ficava parado no centro do mapa foi removido), pela câmera de baixo do drone (a mesma da
missão: RGB 640×480, FOV 90°). Feito para retreinar a YOLO do kit
(`lipo_seg_yolo11.pt`, branch `vision-yolo`), que no simulador confunde o
**reflexo da janela na base azul** com o kit.

## Por que retreinar — o modelo atual neste conjunto (val, conf 0,5, IoU 0,5)

| métrica | valor |
|---|---|
| recall nos kits | **0,36** (142 de 397) |
| precisão | **0,38** (232 falsos positivos) |
| imagens SEM kit com falso positivo | **25 de 70** |

No treino original as métricas eram ~0,99 — a validação não tinha nada parecido
com estes reflexos.

## O que tem aqui

```
data.yaml              pronto para o Ultralytics (nc: 1, names: ['lipo'])
images/train  1614     images/val  418
labels/train           labels/val          (formato YOLO detect)
previews/              amostras com as caixas desenhadas, para conferir
raw/                   os pares originais (vazio/kit) e um meta.json por rodada
generate.log           log da geração
```

- **859 imagens com kit**, 1884 caixas (vários kits por imagem é comum)
- **314 negativos**: a mesma cena SEM kit (bases, reflexos, chão, bloco da
  Fase 4) com arquivo de rótulo vazio — é o que ensina o modelo a não chamar
  reflexo de kit
- **859 cópias aumentadas** (brilho, contraste, tom de cor, ruído, desfoque)
- train/val separados por POSE: a imagem, seu negativo e sua cópia aumentada
  ficam sempre no mesmo lado (sem vazamento entre treino e validação)

## Como variou

| eixo | faixa |
|---|---|
| altura da câmera sobre o kit | 0,4 a 4 m |
| ângulo | yaw do drone 0–360° (o kit aparece girado na imagem), roll/pitch ±15° (vista oblíqua) |
| posição no quadro | qualquer lugar, inclusive cortado na borda |
| onde o kit está | coletas no topo do bloco (1,6 m), entregas (0 a 1,5 m) e chão |
| giro do kit | yaw sorteado de 0 a 360° no spawn |
| luz | 7 rodadas com exposição 8,5–12,5, ISO 800–3200, obturador 30–120; os reflexos mudam com a pose |

## Como os rótulos foram feitos (sem rotular à mão)

Cada pose foi renderizada duas vezes: com a cena **vazia** e depois de **spawnar
os kits**. A diferença entre as duas é o kit; a sombra é separada porque só
escurece a superfície, sem criar bordas. Caixas com lado < 8 px são descartadas
(ruído ou kit longe demais para reconhecer). Pares em que mais de 12% da imagem
mudou foram descartados (render que não repetiu).

**Confira os `previews/` antes de treinar.** Pode haver caixa um pouco maior que
o kit quando a sombra cai sobre algo texturizado.

## Sugestão de treino

Mesmo setup do modelo atual (tirado do checkpoint): `yolo11n.pt`, `imgsz=960`,
`epochs=150`, `batch=16`. **Junte com o dataset real de vocês**, não substitua:
este conjunto só tem o kit do simulador (a bolsa de LiPo do BiguaSim); o real é
que ensina o kit de verdade. Exemplo:

```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=960 epochs=150 batch=16
```

## Limitações (para a próxima rodada)

- **A hora do dia não variou**: `set_day_time`/`set_weather` travam este mundo.
  O sol e a direção dos reflexos são os mesmos em todas as rodadas.
- Sem **orientação** do kit (caixa alinhada aos eixos). Para treinar orientação
  (OBB), dá para tirar o ângulo da mesma máscara de diferença — me peçam.

## Gerar de novo

No repositório `joao_pessoa_2026`:

```bash
scripts/kit_dataset/run_all.sh ~/Documents/kit_dataset_sim/raw 150   # simulador, ~1 h
python3 scripts/kit_dataset/label.py ~/Documents/kit_dataset_sim       # rótulos
```
