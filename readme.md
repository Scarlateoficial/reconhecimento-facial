# Reconhecimento facial, para autenticação em dispositivos IoT

## Criadores:

* #### Autores: Julio Nunes Avelar e Luisa Manoela Romão Salles
* #### Orientador: Paulo Cesar dos Santos
* #### Instituição: [IFSULDEMINAS - Campus Muzambinho](https://muz.ifsuldeminas.edu.br/)


## Pré-requisitos:

* ### Web-Cam

    Voçê precisara de uma web-cam, para que as imagens possam ser coletadas.

* ### Python 3.xx

    Você precisara de um interpletador [Python](https://www.python.org/) para poder testar o projeto.

    Python Anaconda: <https://www.anaconda.com/>
    
    Python site oficial: <https://www.python.org/>

* ### Python Open-CV

    O [Opencv](https://pypi.org/project/opencv-python/) e a biblioteca com todos os recursos para tratamento de imagens que vamos precisar.

* ### Matplotlib

    O [matplotlib](https://matplotlib.org/) e importante para realização dos testes.

* ### Numpy

    O [numpy](https://numpy.org/) sera necessario, para trabalhar com grandes listas e operações matemáticas de forma rapida.

* ### Python Requests

    A biblioteca [requests](https://pypi.org/project/requests/) sera necessaria para consumir a API.

## Instalação:

* Para instalação dos pacotes via Python-pip, utilise o seguinte comando:
``` pip3 install -r utils/requiriment.txt ```

* Para instalação dos pacotes via Anaconda, utilise o seguinte comando:
``` conda install -r utils/requiriment.txt ```

## Execução:

1. Execute o arquivo addnovorosto.py.

    Ao ser executado, o código, ele ira pedir alguns dados de autenticação, depois abrira sua webcam. E em seguida, 100 fotos serão tiradas do seu rosto e salvas com seu ID.

2. Execute o arquivo treino.py.

    Ao ser executado, o código, ira treinar o algoritmo com as novas imagens, e 3 tres arquivos .xml serão gerados.

3. Por fim execute o arquivo authros.py.

    Esse código e responsavel por fazer o reconhecimento facil.

## Diretorios:

* dataset --> Diretorio com as imagens salvas.

* classificadores --> Diretorio com os classificadores e modelos, para o reconhecimento facial.

* core --> Pacote com os arquivos python essenciais, como para autenticação, busca na API e tratamento.

* img --> Imagens uteis, para testes e treinos.

* utils --> Arquivos uteis.

## Arquivos:

* addnovorosto.py --> Responsavel por capturar as imagens e salvar

* authros.py --> Responsavel por fazer o reconhecimento facial

* verificauser.py --> Responsavel por todas as requisições relacionadas aos usuarios na API.

* tratarimg.py --> responsavel por todas as funções de padronização de imagem.

* test.py --> Responsavel pela execução de testes no algoritmo

* treino.py --> Responsavel por treinar o algoritmo

* exemples.py --> Exemplos de algumas funções do Open-CV

* requiriment.txt --> Todas os pacotes python instalados