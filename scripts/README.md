# Scripts

Scripts executáveis do Projeto VACA devem ser pequenos pontos de entrada para funções testáveis em `src/vaca`.

Convenções:

- não escrever em fontes legadas;
- receber caminhos por argumento ou variável de ambiente;
- produzir manifestos de execução;
- falhar explicitamente quando dados/modelos estiverem ausentes;
- nunca substituir automaticamente dados reais por dados sintéticos.

Primeiros scripts previstos:

1. `audit_capture.py`
2. `prepare_cattleeyeview.py`
3. `run_detection_tracking.py`
4. `run_bovine15_pose.py`
5. `build_temporal_dataset.py`
6. `evaluate_pipeline.py`
