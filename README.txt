Instruções:

O arquivo responsável por treinar novos modelos é o UNET_TensorFlow_v3.py. Este modelo está otimizado para não consumir muita memória RAM.
Caso o nome designado ao modelo esteja sendo usado na pasta "models", este será carregado para continuar o treinamento.
O histórico de cada modelo será arquivado na pasta "history" e terá o mesmo nome do modelo.

Os modelos devem ser armazenados na pasta "datasets":
- As imagens originais devem permanecer na pasta "images"
- As mascaras devem permanecer na pasta "masks"

Para utilizar a GPU da máquina, seguir os passos descritos no link abaixo, de acordo com o sistema operacional em uso:
https://www.tensorflow.org/install/pip#step-by-step_instructions

Pesos de redes já treinadas em segmentação podem ser encontrados no link abaixo:
https://pypi.org/project/segmentation-models-pytorch/

O arquivo "requirements.txt" deve estar atualizado com as bibliotecas e suas respectivas versões conforme novas ferramentas sejam criadas

