# VACA

Projeto de visão computacional para análise objetiva da locomoção e da postura de bovinos em vídeo, com foco inicial em câmeras zenitais/top-down.

> **Estado clínico:** pesquisa e desenvolvimento. O projeto não deve emitir diagnóstico veterinário nem score clínico validado de claudicação até existir ground truth revisado por especialistas, validação externa e checkpoint temporal versionado.

## Arquitetura-alvo

```text
captura e auditoria
→ detecção / segmentação
→ tracking por animal
→ pose bovina Bovine15
→ controle de qualidade temporal
→ extração de sinais biomecânicos
→ modelagem temporal
→ avaliação e calibração
→ revisão humana / validação veterinária
```

## Organização

- `src/vaca/` — código novo e modular do Projeto VACA.
- `configs/` — configurações versionadas de detecção, pose, tracking e modelos temporais.
- `scripts/` — pipelines reprodutíveis de preparação, treino e avaliação.
- `tests/` — testes unitários, de integração e de consistência científica.
- `experiments/` — manifestos pequenos e versionados dos experimentos; dados pesados ficam fora do GitHub.
- `docs/` — arquitetura, política de dados, protocolos e decisões técnicas.

## Política de dados

O GitHub contém código, configurações, documentação e manifestos pequenos. Vídeos, imagens, anotações volumosas, checkpoints, runs e outputs permanecem no Google Drive em `VACA_DATA`.

Fontes legadas/originais são tratadas como **somente leitura** durante a migração:

1. CattleEyeView (`dataset/videos` e `dataset/annotation`).
2. `MVP_vaca`.
3. `dados_vaca/LOCAVISAO`.
4. Código ST-GCN histórico já existente neste repositório.

Nenhuma fonte original deve ser sobrescrita por pipelines do VACA.

## Branch de implantação inicial

A reorganização de 2026 está sendo preparada em `project-vaca-2026`, preservando `master` como referência histórica até a migração ser auditada.

## Princípios

1. Reprodutibilidade antes de performance isolada.
2. Nenhum fallback sintético silencioso.
3. Separação explícita entre proposta automática, anotação humana e ground truth clínico.
4. Splits por animal/fazenda quando houver dados clínicos, evitando vazamento entre treino e validação.
5. Toda métrica deve apontar para dataset, configuração e checkpoint versionados.
