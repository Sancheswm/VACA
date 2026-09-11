# VACA V1.2 — Experimental Excellence Baseline — Video 01

Status: `EXPERIMENTAL_NOT_CLINICALLY_VALIDATED`

## Resultado

- Fonte: CattleEyeView `01.mp4`
- Saída final: 1226 frames, 8 FPS, 153.25 s, 960x540, H.264
- Segmentação supervisionada mínima: ExtraTrees em features de aparência + background robusto
- Treino: frames oficiais 1144, 1156 e 1172
- Holdout: 1150 e 1167
- IoU holdout médio: **0.658229**
- Precisão holdout média: **0.858607**
- Recall holdout médio: **0.738229**
- VACA V1 anterior: IoU 0.371
- Ganho relativo de IoU: **+77.42%**

## Correção V1.2

A V1.1 dividia alguns bovinos coerentes em múltiplos tracks por uma heurística de fatiamento da máscara. A V1.2 remove essa prática. Cada componente coerente é tratado como um único candidato; componentes muito largos/grandes, compatíveis com fusão de animais, recebem `GROUP / REVIEW` em vez de identidades fabricadas.

## Overlay

- máscara segmentada por track;
- envelope corporal orientado;
- malha geométrica corporal;
- `COW TRACK` temporário;
- `GROUP / REVIEW` quando há provável fusão;
- confiança de segmentação;
- `MOBILITY PENDING`.

## Guardrails

- track não é identidade permanente da vaca;
- malha é geométrica, não Bovine15;
- não há diagnóstico de claudicação;
- não há BCS;
- mobilidade permanece pendente até RTMPose-Bovine15 + modelo temporal validado.

## Próximo gate

Substituir a segmentação experimental por RTMDet-Ins treinado no dataset canônico e superar IoU 0.658 em holdout apropriado, depois integrar BoT-SORT, Re-ID open-set e RTMPose-Bovine15.
