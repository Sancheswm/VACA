# Arquitetura do Projeto VACA

## Objetivo de engenharia

Construir um pipeline reprodutível para percepção e análise temporal da locomoção bovina em vídeo, priorizando câmeras zenitais/top-down e mantendo separação rígida entre percepção visual, inferência biomecânica e validação clínica.

## Pipeline de referência

1. **Capture audit** — FPS, resolução, duração, distorção, iluminação, perdas e integridade.
2. **Calibration / ROI** — corredor, orientação de marcha e região válida.
3. **Detection / segmentation** — localização do animal e máscara corporal.
4. **Tracking** — identidade persistente e isolamento por indivíduo.
5. **Bovine15 pose** — keypoints anatômicos bovinos com confiança por ponto.
6. **Temporal QC** — oclusão, saltos, keypoints ausentes e consistência de trajetória.
7. **Biomechanical features** — medidas cinemáticas e temporais explicitamente definidas.
8. **Temporal modelling** — modelos sobre séries de keypoints/features; nenhum modelo clínico sem labels adequados.
9. **Evaluation** — métricas por tarefa, splits sem vazamento e calibração.
10. **Human review** — revisão técnica/veterinária e rastreabilidade da evidência.

## Fontes iniciais

- CattleEyeView: vídeos e anotações top-down para detecção, pose e segmentação.
- LOCAVISAO: referência de engenharia para percepção zenital, pré-anotação, Bovine15 e hand-off temporal.
- MVP_vaca: componentes e hipóteses legadas a serem auditados antes de reutilização.
- ST-GCN histórico deste repositório: referência legada de modelagem temporal, não baseline clínico automático.

## Regra de promoção

Um componente só entra no pipeline principal após possuir:

- entrada e saída documentadas;
- teste mínimo automatizado;
- configuração versionada;
- evidência de execução em dados reais;
- ausência de fallback sintético silencioso;
- métricas compatíveis com a tarefa que ele declara resolver.

## Regra clínica

Nenhum output deve ser chamado de diagnóstico, classificação clínica de claudicação ou equivalente enquanto não houver ground truth veterinário revisado, desenho de validação apropriado e avaliação independente.
