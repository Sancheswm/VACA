# VACA V1 — Arquitetura Definitiva baseada em auditoria SOTA 2024–2026

Status: **Architecture Decision Record / baseline oficial para implementação**  
Data: 2026-09-11  
Branch: `project-vaca-2026`

## 1. Objetivo

Construir uma plataforma de visão computacional para bovinos com capacidade funcional da classe do CattleEye: captura zenital 2D, identificação individual, mobilidade/claudicação, body condition score (BCS), perfil longitudinal, alertas, evidência auditável, operação edge/offline e integração por API.

A V1 não deve copiar a arquitetura proprietária do CattleEye. Deve reproduzir e, quando possível, superar as capacidades publicamente documentadas por meio de uma implementação independente, cientificamente validada e com componentes cujo licenciamento seja compatível com produto comercial.

## 2. Regra clínica não negociável

Nenhum componente de percepção, pose, Re-ID ou modelagem temporal autoriza sozinho a emissão de um diagnóstico veterinário. Até que existam ground truth de especialistas, validação por animal e por fazenda, avaliação externa e checkpoint temporal versionado, a saída deve ser tratada como `PENDING_HUMAN_REVIEW`, `NOT_CLINICALLY_VALIDATED` ou equivalente.

## 3. Benchmark externo: CattleEye

O CattleEye é o benchmark de produto e validação, não um template de código. Evidências públicas relevantes:

- câmera 2D sobre a saída da ordenha e monitoramento contínuo de mobilidade/BCS;
- validação de mobilidade em 11 fazendas, 42 sessões e 40.116 pares entre escores humanos e automatizados (J. Dairy Sci., 2025; DOI 10.3168/jds.2024-25940);
- validação de BCS 2D com grande conjunto de avaliações manuais e modelo ordinal EfficientNetV2 (J. Dairy Sci., 2024; DOI 10.3168/jds.2023-23894);
- estudo de identificação: 87/103 vacas identificadas em 7 dias e 86/87 identificações corretas entre as vacas identificadas, mostrando que cobertura e tempo de enrolamento são tão importantes quanto precisão;
- ensaio clínico randomizado de 2026 com 419 vacas, demonstrando impacto de alertas automatizados integrados ao manejo (J. Dairy Sci., 2026; S0022-0302(26)03060-2).

## 4. Fontes de dados VACA

### Read-only / legado

- CattleEyeView: 14 vídeos top-down e anotações de detecção, pose e segmentação.
- `MVP_vaca`: componentes de detecção, pose, telemetria e heurísticas anteriores.
- `dados_vaca/LOCAVISAO`: pipeline de percepção/preanotação, Bovine15, evidência e integração OpenMMLab.
- código ST-GCN histórico do branch `master`.

### VACA_DATA

Todos os novos dados, pesos, execuções, ground truth, relatórios e manifests ficam em `VACA_DATA` no Google Drive, nunca diretamente no GitHub.

## 5. Arquitetura V1 escolhida

### Etapa 0 — Capture Quality Gate

**Modelo:** sem rede pesada como padrão. OpenCV + estatística + regras calibradas.

Entradas e verificações:
- blur/foco (Laplaciano ou métrica equivalente);
- exposição/histograma;
- FPS real, frames perdidos e timestamp;
- ocupação da ROI do corredor;
- deslocamento da câmera;
- homografia/calibração do corredor;
- direção de passagem;
- oclusão e qualidade mínima da passagem.

Saída: `PASS`, `REVIEW` ou `REJECT`, com métricas de qualidade anexadas ao passage manifest.

### Etapa 1 — Detecção + segmentação de instância

**Produção V1: RTMDet-Ins-S ou RTMDet-Ins-M, MMDetection.**

Motivos:
- real-time instance segmentation;
- integração nativa com ONNX/TensorRT via ecossistema OpenMMLab;
- licença Apache-2.0 do ecossistema;
- máscara é útil para BCS, Re-ID, eixo corporal e características de silhueta.

**Challenger:** Ultralytics YOLO11/YOLO26 Seg.

Restrição: Ultralytics permanece apenas benchmark/research enquanto não houver licença Enterprise ou decisão explícita de tornar o produto compatível com AGPL-3.0. A documentação oficial da Ultralytics exige Enterprise para produto comercial/proprietário, uso privado e edge fechado.

Métricas de promoção:
- box AP e mask AP;
- recall de vaca por passagem;
- erro por tamanho/posição/oclusão;
- latência no hardware edge alvo;
- robustez por fazenda/câmera.

### Etapa 2 — Tracking de passagem

**Produção V1: BoT-SORT adaptado ao corredor fixo.**

Fusão:
- motion prediction;
- IoU/máscara;
- appearance embedding curto;
- regras de direção/ROI do corredor.

**Fallback:** ByteTrack.

Importante: `track_id` é efêmero e jamais deve ser tratado como identidade permanente do animal.

### Etapa 3 — Identificação permanente / Cow Re-ID

**Produção V1 baseline: ResNet50 DML, embedding 128-D, cosine distance, Triplet Margin Loss, semi-hard mining.**

Essa configuração é sustentada por estudo de 2025 de melhores práticas em identificação bovina por padrão de pelagem, com resultados muito altos em OpenCows2020 e BeefCattle2024.

**Obrigatório:** protocolo open-set.

O sistema deve possuir classe `UNKNOWN`; nunca deve forçar um ID quando a distância/uncertainty não sustentar a associação.

Galeria por animal:
- múltiplas passagens de alta qualidade;
- medoid/centroid embedding + distribuição;
- atualização conservadora apenas após confirmação;
- fusão opcional de mask/body-shape;
- integração com RFID/ID da ordenha durante enrolamento quando disponível.

**Challenger V1.1:** OpenCowID (WACV 2026), que demonstrou zero-shot/open-set Re-ID com síntese de padrões de pelagem e centroid-guided feature learning.

Métricas:
- CMC@1;
- mAP;
- false match rate;
- false unknown rate;
- ID coverage;
- tempo de enrolamento;
- desempenho em vacas nunca vistas, dias diferentes e câmeras/fazendas diferentes.

Meta de produto: alta precisão entre IDs aceitos **e** cobertura próxima da totalidade do rebanho no menor tempo possível; precisão isolada não é suficiente.

### Etapa 4 — Pose bovina Bovine15

**Produção V1: RTMPose-M (ou RTMPose-S no edge) via MMPose, fine-tuned para Bovine15.**

Bovine15 será o schema canônico e versionado. Antes do treinamento definitivo, os nomes anatômicos, conexões, simetrias, índices, visibility flags e convenção de coordenadas devem ser congelados em uma especificação própria.

Treino:
- CattleEyeView para transferência top-down onde o schema permitir;
- anotações VACA revisadas em Bovine15;
- dados reais de fazenda;
- hard examples de oclusão e múltiplos animais.

**Challengers:**
- RTMPose + neck multiescala específico;
- DeepLabCut/SuperAnimal apenas em pesquisa/benchmark;
- KITPose/ViTPose como benchmarks de precisão.

Licenciamento/proveniência:
- AP-10K é explicitamente indicado pelo MMPose para uso não comercial; não deve tornar-se dependência silenciosa do modelo comercial;
- SuperAnimal pretrained models são indicados pelo DeepLabCut como research/non-commercial use. Podem ser avaliados, mas não promovidos ao núcleo comercial sem revisão de licença.

Métricas:
- OKS-mAP;
- PCK;
- erro por keypoint;
- jitter temporal;
- continuidade;
- taxa de keypoints válidos;
- desempenho em oclusão;
- holdout por animal e por fazenda.

### Etapa 5 — Normalização biomecânica

Antes do modelo temporal, cada sequência será transformada para reduzir dependência de câmera e escala:

- correção/homografia do corredor;
- orientação pelo heading do animal;
- escala pelo comprimento corporal ou referência anatômica robusta;
- centralização em eixo corporal;
- interpolação apenas de gaps curtos com flag de imputação;
- resampling temporal para T fixo;
- preservação explícita do confidence score.

Canais temporais sugeridos:
- joints `(x, y, confidence)`;
- bone vectors;
- first derivative / velocity;
- second derivative / acceleration;
- visibility/imputation mask.

### Etapa 6 — Features biomecânicas explicáveis

Features auxiliares, não diagnóstico isolado:
- curvatura e oscilação lateral da coluna;
- assimetria de ombros/quadril;
- oscilação de cabeça/pescoço;
- deslocamento relativo dos cascos;
- fase entre membros;
- stride proxy;
- stance/swing duration;
- cadence;
- velocidade/aceleração;
- curvatura da trajetória;
- simetria temporal e espacial.

Uma única medida (ex.: ângulo dorsal) é insuficiente e não será promovida como classificador final.

### Etapa 7 — Modelo temporal de mobilidade

**Produção V1 candidato principal: CTR-GCN / ST-GCN++ sobre Bovine15.**

Motivos:
- forte inductive bias para esqueleto;
- modela relações espaciais e temporais explicitamente;
- tende a ser mais apropriado que Transformers muito grandes quando os dados clínicos são limitados;
- integra-se ao legado ST-GCN existente e ao ecossistema MMAction2.

Streams candidatos:
1. joint;
2. bone;
3. joint motion;
4. bone motion;
5. confidence/visibility.

**Baseline obrigatório A:** BLSTM com três camadas de 128 unidades. Em 2026, um estudo bovino com keypoints T-LEAP mostrou 85% de acurácia e desempenho superior a SVM com features manuais, inclusive usando ~1 s de vídeo.

**Baseline obrigatório B:** Random Forest/XGBoost sobre features biomecânicas explícitas. Um estudo de 2025 reportou 93,8% de acurácia com fusão de sete características de movimento, embora a geometria exata de captura desse trabalho deva ser verificada antes de extrapolação zenital.

**Challenger:** SkateFormer. É estado da arte em skeleton action recognition humano, mas não será adotado como produção antes de demonstrar vantagem em bovinos e generalização por fazenda.

**Challenger opcional:** PoseC3D / RGB temporal branch.

Saídas:
- continuous mobility risk score;
- probability calibrated;
- ordinal mobility head;
- binary lame/non-lame head apenas após validação;
- uncertainty;
- quality flag;
- evidência de passagem.

Mild/borderline cases devem poder retornar `REVIEW`, não ser forçados em classe clínica.

### Etapa 8 — Body Condition Score

**Produção V1: EfficientNetV2-S + ordinal regression, dorsal standardized ROI.**

Base científica: arquitetura semelhante foi usada na validação 2D do CattleEye; o enquadramento ordinal preserva a natureza ordenada do BCS.

Entrada:
- mask/crop dorsal normalizado;
- somente frames aprovados pelo quality gate;
- múltiplos frames por passagem.

Agregação:
- mediana/estimador robusto por passagem;
- agregação longitudinal por dia/semana.

Saídas:
- expected BCS;
- confidence/uncertainty;
- delta BCS;
- trend.

**Challenger:** ConvNeXt-Tiny continuous regression + ordinal consistency. Trabalho de 2026 em vista lateral, com 3.208 imagens de 211 vacas e ground truth de três avaliadores, mostrou MAE 0,41; deve ser benchmarkado, mas não substitui evidência top-down.

### Etapa 9 — Perfil longitudinal e alerta

V1 deve começar com regras transparentes sobre séries temporais, antes de um modelo profundo de alertas:

- daily passage aggregation;
- EWMA / robust moving averages;
- change-point detection;
- persistência;
- rate-of-change;
- score absoluto;
- confiança/qualidade;
- cooldown e deduplicação de alertas.

Cada alerta deve guardar:
- cow_id;
- score e tendência;
- confidence;
- passagem responsável;
- frames/clip;
- mask;
- skeleton overlay;
- features principais;
- versão de todos os modelos;
- hashes/manifests.

Os thresholds do ensaio do CattleEye podem servir como referência conceitual, mas não devem ser copiados como thresholds clínicos VACA sem calibração independente.

### Etapa 10 — Edge + cloud

Pipeline edge:

`capture -> quality -> segmentation -> tracking -> ReID -> pose -> mobility -> BCS -> local event store -> sync`

Requisitos:
- offline-first;
- fila persistente local;
- inferência ONNX Runtime/TensorRT quando possível;
- retomar upload/sync sem duplicar eventos;
- observabilidade e versionamento de modelos;
- API-first.

Dashboard/cloud não deve recalcular inferência primária como regra; deve receber eventos versionados, histórico e evidências, permitindo reprocessamento auditável quando necessário.

## 6. Stack de software escolhida

| Bloco | V1 principal | Challenger | Nota |
|---|---|---|---|
| Detecção/segmentação | RTMDet-Ins-S/M + MMDetection | YOLO11/26-Seg | Ultralytics requer decisão/licença comercial |
| Tracking | BoT-SORT custom | ByteTrack | track_id não é cow_id |
| Cow Re-ID | ResNet50-DML 128-D | OpenCowID | Open-set obrigatório |
| Pose | RTMPose-M/S + MMPose | KITPose / DLC research | Bovine15 próprio |
| Mobility temporal | CTR-GCN / ST-GCN++ | SkateFormer / PoseC3D | BLSTM é baseline obrigatório |
| Explainable gait head | RF/XGBoost features | MLP | nunca usar feature única |
| BCS | EfficientNetV2-S ordinal | ConvNeXt-T regression | top-down VACA precisa de validação própria |
| Preannotation | SAM 2 | geometry/GrabCut | anotação, não inferência clínica principal |
| Runtime | PyTorch train; ONNX/TensorRT deploy | — | edge-first |

## 7. Licenciamento — decisões

### Preferidos no núcleo comercial

- MMDetection / RTMDet: ecossistema OpenMMLab, Apache-2.0.
- MMPose / RTMPose: Apache-2.0.
- MMAction2: Apache-2.0.
- SAM 2: Apache-2.0 para modelos/código principal, uso prioritário em anotação.
- BoT-SORT: verificar e registrar licença do commit exato antes de vendorização.

### Pesquisa/benchmark com restrição

- Ultralytics YOLO: AGPL-3.0 ou Enterprise. Sem Enterprise, não promover para produto fechado.
- AP-10K data: MMPose informa non-commercial use only.
- DeepLabCut SuperAnimal weights: documentação atual informa research/non-commercial use.

Todo dataset/checkpoint deverá possuir um `provenance.json` com licença, DOI/URL, versão, hash e permissões de uso.

## 8. Datasets externos prioritários

| Dataset | Uso VACA |
|---|---|
| CattleEyeView | top-down detection/pose/segmentation/tracking; já no Drive |
| OpenCows2020 | Re-ID/top-down benchmark |
| BeefCattle2024 | Re-ID cross-breed/view benchmark |
| MultiCamCows2024 | Re-ID cross-camera/day benchmark, se licença permitir |
| BLSTM lameness trajectories (Russello et al. 2026) | baseline temporal externo |
| AnimalPose | pose transfer/benchmark, sujeito a licença |
| AP-10K | pesquisa/benchmark; não usar como dependência comercial sem autorização |

## 9. Split e anti-leakage

É proibido medir desempenho clínico com random frame split como evidência principal.

Hierarquia de avaliação:
1. frame-level perception;
2. passage-level;
3. cow-level holdout;
4. camera/farm-level holdout;
5. external farm validation;
6. prospective longitudinal cohort;
7. lesion association;
8. prospective intervention study.

Regras:
- frames da mesma passagem não podem cruzar train/test;
- vacas de teste não podem aparecer no treino do modelo clínico quando o objetivo for generalização por animal;
- pelo menos uma fazenda/câmera deve permanecer totalmente externa na validação final;
- Re-ID deve ser testado em open-set e cross-day/cross-camera;
- reportar intervalos de confiança e calibration, não apenas accuracy/F1.

## 10. Métricas oficiais

### Percepção
- box AP / mask AP;
- recall por passagem;
- latência;
- failure rate.

### Pose
- OKS-mAP;
- PCK;
- error por keypoint;
- continuity/jitter;
- visibility success rate.

### Re-ID
- CMC@1;
- mAP;
- false match rate;
- false unknown rate;
- coverage;
- time-to-enrolment.

### Mobilidade
- AUROC/AUPRC;
- sensitivity/specificity;
- balanced accuracy/F1;
- weighted kappa;
- Gwet AC1/AC2 quando pertinente;
- confusion por grau;
- sensitivity específica para casos leves;
- calibration/Brier/ECE;
- lesion association.

### BCS
- MAE;
- weighted kappa;
- exact agreement;
- ±0.25 e ±0.50 agreement;
- Bland–Altman;
- Delta-BCS error;
- repeatability.

## 11. Gates para promoção

Nenhum modelo substitui o anterior apenas por maior score agregado.

Um challenger só é promovido se demonstrar vantagem estatística e operacional em:
- holdout por animal;
- holdout por fazenda/câmera;
- calibration;
- casos leves/borderline;
- latência/memória;
- estabilidade temporal;
- licença/proveniência;
- interpretabilidade/evidência.

## 12. Ordem de implementação

Fase A: capture audit + RTMDet-Ins + manifests.  
Fase B: tracking + crop/mask canônico.  
Fase C: Bovine15 RTMPose + benchmark.  
Fase D: Re-ID open-set + enrolment.  
Fase E: dataset clínico temporal e features biomecânicas.  
Fase F: BLSTM + RF + CTR-GCN/ST-GCN++ benchmark.  
Fase G: BCS ordinal + longitudinal profile.  
Fase H: edge runtime, API, dashboard e alert engine.  
Fase I: external validation + prospective study.

## 13. Decisões rejeitadas

- **Um único ângulo dorsal como diagnóstico:** rejeitado; pode permanecer como feature explicável.
- **Re-ID closed-set forçado:** rejeitado; `UNKNOWN` obrigatório.
- **Random frame split como validação:** rejeitado.
- **Ultralytics como dependência comercial automática:** rejeitado sem licença Enterprise/decisão AGPL.
- **AP-10K/SuperAnimal como pesos comerciais silenciosos:** rejeitado por restrições declaradas de uso.
- **SkateFormer diretamente em produção:** rejeitado até superar GCN/BLSTM no domínio bovino.
- **Depth camera obrigatória na V1:** rejeitado para manter paridade de baixo custo com câmera RGB 2D; depth permanece opção futura.
- **Score clínico antes de ground truth veterinário:** rejeitado por design.

## 14. Conclusão de arquitetura

A VACA V1 será uma arquitetura modular e auditável de **segmentação + tracking + open-set Re-ID + Bovine15 pose + graph temporal mobility + ordinal BCS + longitudinal alerting**, com processamento edge/offline e evidência por alerta.

A escolha inicial de produção é deliberadamente conservadora e comercialmente orientada: **RTMDet-Ins → BoT-SORT → ResNet50-DML → RTMPose-Bovine15 → CTR-GCN/ST-GCN++ → EfficientNetV2 ordinal**. Cada bloco possui um challenger explícito e um protocolo de promoção baseado em evidência.

## 15. Referências-chave auditadas

- Siachos N. et al. Development and validation of a fully automated 2-dimensional imaging system generating body condition scores for dairy cows using machine learning. Journal of Dairy Science, 2024. DOI: 10.3168/jds.2023-23894.
- Siachos N. et al. Evaluation of a fully automated 2-dimensional imaging system for real-time cattle lameness detection using machine learning. Journal of Dairy Science, 2025. DOI: 10.3168/jds.2024-25940.
- Randomized controlled trial evaluating automated 2D imaging for lameness control. Journal of Dairy Science, 2026, article S0022-0302(26)03060-2.
- Wu S. et al. A top-down deep neural network for multi-dairy cows pose estimation and lameness detection. Computers and Electronics in Agriculture, 2025. DOI: 10.1016/j.compag.2025.110911.
- Russello H. et al. Lameness detection in dairy cows using pose estimation and bidirectional LSTMs. Smart Agricultural Technology, 2026. DOI: 10.1016/j.atech.2026.101831.
- Deep metric learning for individual cattle identification using coat patterns: Proposal for a best practice. Computers and Electronics in Agriculture, 2025. DOI: 10.1016/j.compag.2025.110754.
- Prabhune O., Kim Y. OpenCowID: Zero-Shot Visual Identification of Dairy Cows. WACV 2026.
- Ye S. et al. SuperAnimal pretrained pose estimation models for behavioral analysis. Nature Communications 15, 5165 (2024).
- Do J., Kim M. SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition. ECCV 2024 proceedings.
- OpenMMLab MMPose, MMDetection/RTMDet, MMAction2 official repositories/documentation.
- AnimalEyeQ CattleEyeView official repository/dataset.
- Ultralytics current licensing documentation (AGPL-3.0 / Enterprise).
