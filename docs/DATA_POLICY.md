# Política de dados — VACA

## Fonte da verdade

- **GitHub:** código, documentação, configurações, testes e manifestos pequenos.
- **Google Drive / VACA_DATA:** dados brutos, anotações volumosas, dados processados, checkpoints, runs, outputs, ground truth clínico, relatórios e exportações.

## Estrutura oficial no Drive

```text
VACA_DATA/
├── 00_MANIFESTS/
├── 01_RAW/
├── 02_ANNOTATIONS/
├── 03_PROCESSED/
├── 04_CHECKPOINTS/
├── 05_RUNS/
├── 06_OUTPUTS/
├── 07_CLINICAL_GROUND_TRUTH/
├── 08_REPORTS/
├── 09_EXPORTS/
└── 99_ARCHIVE/
```

## Fontes legadas/originais

As seguintes fontes são consideradas somente leitura durante a implantação:

- `dataset/videos`
- `dataset/annotation`
- `MVP_vaca`
- `dados_vaca/LOCAVISAO`

Pipelines novos não devem gravar dentro dessas fontes.

## Convenção de runs

Cada execução relevante deve produzir um identificador único, por exemplo:

`YYYYMMDD_HHMMSS__task__model__dataset`

O run deve registrar, quando aplicável:

- commit Git;
- arquivo de configuração;
- origem dos dados;
- split usado;
- seed;
- checkpoint de entrada/saída;
- métricas;
- artefatos gerados;
- status (`SUCCESS`, `FAILED`, `PENDING_REVIEW`).

## Dados clínicos

Ground truth clínico deve permanecer separado de labels puramente visuais. A origem do rótulo, avaliador, protocolo, data e unidade experimental devem ser rastreáveis. Dados clínicos sensíveis não devem ser publicados no repositório público.
