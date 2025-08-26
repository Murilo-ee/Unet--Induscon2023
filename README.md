# Segmentação de Pólipos com U-Net (Induscon 2023)

Projeto para **segmentação semântica** de pólipos em imagens endoscópicas usando **U-Net**. 
Os datasets utilizados são CVC-ClinicDB e Kvasir-SEG.
---

## Objetivo
- Treinar e avaliar uma **U-Net** para segmentar pólipos.
- Disponibilizar um **pipeline completo** (preparação → treino → validação → inferência → análise).
- Padronizar **artefatos** (pesos, logs, métricas e máscaras preditas) para auditoria e comparação.

---

## Descrição
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
  - Lote de imagens em `TEST_IMAGES/` → salvamento de máscaras e métricas por amostra.

---


## Estrutura

| Caminho | Função |
|---|---|
| `UNET_TensorFlow_v3.py` | Treino U-Net (TF/Keras) + callbacks |
| `UNET_TensorFlow_v2.py` | Versão anterior da U-Net |
| `apply_model.py` | Inferência em lote (gera máscaras preditas) |
| `check_model_performance.py` | Métricas em `TEST_IMAGES` |
| `complement_functions.py` | Utilitários: métricas, I/O, mascaramento |
| `convert_format_and_size.py` | Conversão de formato/tamanho de imagens |
| `get_information_about_datasets.py` | Inspeção/contagem de imagens e máscaras |
| `output_article.py` | Geração de figuras/relatórios |
| `preprocess_pipeline.py` | Remoção de reflexos, normalização de iluminação, patchify |
| `preprocessing_images.py` | Preparação em lote (pré-processamento) |
| `segmentation_models_pytorch.py` | Utilidades SMP (arquiteturas/pesos) |
| `video_frames_extractor.py` | Extração de frames de vídeos |
| `requirements.txt` | Dependências |
| `README.txt` | Instruções originais de uso/treino |
| `models/` | Modelos treinados (pesos) |
| `pretrained_models/` | Pesos pré-treinados (quando aplicável) |
| `history/` | Logs CSV do treinamento |
| `output_midia/` | Figuras/plots/resultados |
| `tmp/` | Arquivos temporários |



---

## Tecnologias
- **TensorFlow/Keras** (modelagem e treino da U-Net)
- **OpenCV** e **PIL** (pré/pós-processamento de imagens)
- **NumPy**, **pandas**, **matplotlib** (ETL leve, logs e gráficos)
- **TensorBoard** (monitoramento)
- **patchify**, utilitários **PyTorch/torchvision** usados em partes do pré-processamento

