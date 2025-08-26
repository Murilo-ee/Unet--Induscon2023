# Segmentação de Pólipos com U-Net (Induscon/LA-CCI 2023)

Projeto para **segmentação semântica** de pólipos em imagens endoscópicas usando **U-Net**. O repositório organiza **pré-processamento**, **modelagem**, **treinamento**, **pós-processamento** e **avaliação** em módulos independentes, visando reprodutibilidade e fácil adaptação.

---

## Objetivo
- Treinar e avaliar uma **U-Net** para segmentar pólipos.
- Disponibilizar um **pipeline completo** (preparação → treino → validação → inferência → análise).
- Padronizar **artefatos** (pesos, logs, métricas e máscaras preditas) para auditoria e comparação.

---

## Principais funcionalidades
- **Pré-processamento**:
  - Remoção de **reflexos especulares** e ruídos com OpenCV (grayscale + limiar adaptado, dilatação e **inpainting**).
  - **Normalização de iluminação** (morphological operations).
  - **Crop** e **resize** para **256×256**; opção de **patches** (patchify) quando aplicável.
- **Modelagem (U-Net)**:
  - Encoder–decoder com **skip connections**, convoluções 3×3, **BatchNorm**, **ReLU**, **dropout** e **upsampling** por transposed conv.
  - Saída sigmoide para **máscara binária**.
- **Treinamento**:
  - **Loss**: Dice Loss.
  - **Otimizador**: Adam.
  - **Callbacks**: CSVLogger (histórico por época), ModelCheckpoint (melhor validação), EarlyStopping (paciência), TensorBoard.
- **Pós-processamento**:
  - **Opening** (erosão→dilatação) para limpeza de pontos isolados na máscara.
  - **Binarização** com limiar final.
- **Avaliação**:
  - Métricas implementadas: **SSIM**, **Jaccard/IoU**, **Precision**, **Recall** (e utilitários para **MSE**, Dice).
  - Relatórios e gráficos consolidados (curvas de aprendizado; sumários em CSV).
- **Inferência**:
  - Lote de imagens em `TEST_IMAGES/` → salvamento de **máscaras preditas** e métricas por amostra/conjunto.

---

## Estrutura

UNET_TensorFlow_v3.py # definição e treino da U-Net (TF/Keras) + callbacks
check_model_performance.py # inferência e cálculo de métricas em TEST_IMAGES
complement_functions.py # utilitários de métricas, IO e mascaramento
preprocess_pipeline.py # remoção de reflexos, normalização de iluminação, patchify
preprocessing_images.py # pipeline de preparação em lote
README.txt # instruções originais de uso/treino


---

## Tecnologias
- **TensorFlow/Keras** (modelagem e treino da U-Net)
- **OpenCV** e **PIL** (pré/pós-processamento de imagens)
- **NumPy**, **pandas**, **matplotlib** (ETL leve, logs e gráficos)
- **TensorBoard** (monitoramento)
- **patchify**, utilitários **PyTorch/torchvision** usados em partes do pré-processamento

