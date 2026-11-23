# Identificador de Frutas com IA

Projeto acadêmico FAMETRO: seis classes (maçã, banana, laranja, limão,
abacaxi e morango) com TensorFlow/Keras, MobileNetV2 e Streamlit.

## Executar o aplicativo

No PowerShell, na pasta do projeto, use o ambiente existente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Envie JPG, JPEG, PNG ou WEBP (até 5 MB). A interface mostra a imagem, as seis
probabilidades e avisa se a maior estiver abaixo de 70%. Esse limiar é uma
heurística, não uma garantia estatística. Sem modelo válido, não há substituição
pelo ImageNet genérico. Erros técnicos são registrados no terminal.

## Preparar, treinar e avaliar

```powershell
# Já executado: não repita se dataset_frutas6 existir.
.\.venv\Scripts\python.exe preparar_dataset.py
# Gera o modelo; não sobrescreve resultados existentes.
.\.venv\Scripts\python.exe treinar_modelo.py
# Pode ser executado novamente para conferir o modelo salvo.
.\.venv\Scripts\python.exe avaliar_modelo.py
```

Não é necessário ativar o ambiente nem reinstalar dependências. Para uma nova
experiência, use `treinar_modelo.py --output model_experimento`; avalie com
`avaliar_modelo.py --model-dir model_experimento --output training/experimento`.
O app usa apenas os caminhos em `settings.py`.

## Dados e método

O Fruits-360 original permanece em `../dataset_fruits360`. A preparação copia
800 imagens de treino e 250 de teste por classe: 4.800 + 1.500 = 6.300.
Usa semente 42, verifica e decodifica cada imagem selecionada com Pillow,
confere a cópia por SHA-256 e impede duplicatas exatas entre os conjuntos.
Não apaga pastas existentes. `dataset_frutas6/manifest.json` registra as origens.
Não inclui Orange peeled, Strawberry Wedge, pimentões ou tomates. Lemon inclui
Lemon Meyer, mas não Limes. A amostra é equilibrada por fruta, não por variedade.

O treinamento reserva 20% por classe para validação (3.840 treino e 960 validação).
O teste não participa do ajuste. A MobileNetV2 fica congelada; as características
são extraídas uma vez em lotes de 16, usando imagens de 160×160. Apenas os
vetores compactos ficam na memória. A cabeça usa Dropout, regularização L2 e
parada antecipada pela perda de validação, restaurando os melhores pesos.
Não há fine-tuning nem aumento das imagens nesta primeira versão.
Os pesos ImageNet locais são reutilizados; se não existirem no cache, Keras
precisa obtê-los. Não é necessário baixar novamente o Fruits-360.

Treino, avaliação e app compartilham redimensionamento bilinear e conversão
RGB. A normalização está dentro do modelo salvo. A ordem das saídas vem de
`model/classes.json`, sem `decode_predictions()`.

## Arquivos gerados

- `model/fruit_model.keras`: rede completa, pronta para inferência.
- `model/classes.json`: classes e pré-processamento.
- `model/training.json`: histórico real, época selecionada e arquivos de treino/validação.
- `training/execucao.log`: saída da execução local inicial.
- `training/evaluation/metrics.json`: acurácia, entropia cruzada, precisão, recall, F1 e matriz por classe.
- `training/evaluation/predictions.csv`: resultado individual de cada imagem de teste.
- `training/evaluation/confusion_matrix.png`: linhas reais e colunas previstas.

## Limitações

O Fruits-360 usa imagens frequentemente padronizadas, com fundo branco.
Imagens sequenciais podem ser muito semelhantes, inclusive na validação
aleatória; remover duplicatas exatas não elimina essa correlação. Os números
obtidos medem este conjunto, não o desempenho em fotografias do cotidiano.
Fotos externas ainda precisam de avaliação separada com rótulos conhecidos.
O modelo sempre distribui a probabilidade entre as seis classes; não foi treinado
para detectar objetos fora dessas categorias ou várias frutas na mesma imagem.

## Ambiente

Python 3.12, TensorFlow 2.16.2, NumPy 1.26.4, Pillow 10.4.0 e Streamlit 1.41.1.
A execução local detectou somente CPU; a presença de RTX 3050 não significa
que o TensorFlow nativo do Windows use a GPU. Não foram alteradas dependências.

## Resultado da execução local inicial

O treinamento em CPU levou aproximadamente 127 segundos. Foram usadas 3.840
imagens para ajuste e 960 para validação. Na avaliação independente de 1.500
imagens do Fruits-360, houve 1.499 acertos (99,93%) e entropia cruzada de
0,001096. Uma maçã foi prevista como limão. Com o limiar predefinido de 70%,
essa imagem ficou abaixo da confiança mínima: 1.499 resultados aceitos, todos
corretos neste teste. Esse resultado não representa validação em fotos reais.

Verificações adicionais: leitura do modelo salvo, uma imagem de cada classe,
igualdade do pré-processamento entre avaliação e serviço, abertura do Streamlit,
exibição do resultado, aviso de baixa confiança, upload inválido e modelo ausente.

## Testes externos de 23/11/2026

Os 11 arquivos de Downloads/Imagens-testes foram avaliados e preservados em
`training/testes_externos/2026-11-23_rodada02_completa`.
Resultado: 5 acertos top-1, 3 erros aceitos com confiança >= 70% e 3 resultados
inconclusivos. As previsões dos inconclusivos também estavam erradas.
Esta amostra pequena tem duas versões da mesma foto de morangos e inclui
as seis falhas da rodada01; não somar os totais das rodadas.
Consulte `LEIA-ME.txt`, `registro.json` e `resultados.csv` nessa pasta.
Não confundir esses resultados com a avaliação no Fruits-360.

Para repetir com outros rótulos ou modelo atual, use um novo destino:

```powershell
.\.venv\Scripts\python.exe avaliar_fotos_externas.py --imagens "$env:USERPROFILE\Downloads\Imagens-testes" --rotulos training/rotulos_fotos_externas.csv --output training/testes_externos/nova_rodada
```

## Interface revisada

A tela organiza envio e resultado em duas etapas, com estados de espera,
foto pronta, análise, sugestão e baixa confiança. A prévia preserva a proporção
e tem altura limitada. As seis pontuações ficam em um único detalhamento.
O resultado usa “Sugestão do modelo” e “Pontuação”, sem prometer acerto.
Trocar ou remover a imagem descarta a previsão anterior. O layout empilha
os painéis em telas menores e usa controles nativos de teclado e foco visível.
Validado com AppTest nos estados inicial, resultado e inconclusivo, e no
navegador com upload, análise e layout de 390 px. Modelo e métricas não foram alterados.

## Rodada de 30 fotos obtidas na internet

Em 23/11/2026, o modelo inicial foi avaliado em cinco fotos por fruta, do
Wikimedia Commons. Acertos top-1: maçã 4/5, banana 5/5, laranja 0/5,
limão 2/5, abacaxi 0/5 e morango 5/5; total 16/30 (53,33%).
Na interface com limiar de 70%: 15 acertos aceitos, 7 erros aceitos e 8
inconclusivos. Um inconclusivo tinha a classe correta como primeira opção.
Limão inclui dois amarelos (2/2 corretos) e três verdes (0/3 corretos).
Amostra pequena de conveniência; algumas composições/sessões são relacionadas.
Seleção encerrada antes de consultar as previsões; sem novo treinamento.
Relatório com imagens e fontes em
`training/testes_externos/2026-11-23_internet30/relatorio.html`.
Nessa pasta também estão protocolo, fontes/licenças, CSV, JSON, código usado
e uma cópia do modelo. Não misturar os números com o teste do Fruits-360.

## Instalar em outro computador a partir do GitHub

O repositório inclui o modelo treinado em `model/`; não é preciso treinar
para abrir o aplicativo. Instale Python 3.12 e Git e execute no PowerShell:

```powershell
git clone https://github.com/jaovcodes/identificador-frutas-ia.git
cd identificador-frutas-ia
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O repositório é público e pode ser clonado sem convite.
Depois da instalação inicial, basta executar o último comando ou abrir
`executar.bat`. O dataset original e `dataset_frutas6` não são versionados.
Para reproduzir o treinamento, obtenha o Fruits-360 separadamente e informe
sua pasta com `preparar_dataset.py --origem CAMINHO`.

As 30 fotos de avaliação do Commons têm fontes, autores e licenças individuais
em `training/testes_externos/2026-11-23_internet30/fontes.json` e `relatorio.html`.
Essas licenças se aplicam às respectivas imagens, não automaticamente ao código.
As evidências das primeiras rodadas incluem fotos, prints e registros. A origem
e as licenças dessas fotos iniciais ainda não foram documentadas; não atribua
a elas as licenças das fotos do Commons. Os logs de treinamento, avaliação e
verificação também estão versionados. Ambientes virtuais, datasets, logs do
servidor e cópias redundantes do modelo permanecem locais.
Relatórios técnicos incluem caminhos da execução original;
esses caminhos são históricos, não configuração necessária para instalar o app.

## Evidências para o artigo e a apresentação

- [Índice e orientação de leitura](documentacao/INDICE_EVIDENCIAS.txt)
- [Relatório técnico consolidado](documentacao/RELATORIO_TECNICO_RESULTADOS_2026-11-23.txt)
- [Pacote de evidências para download](documentacao/EVIDENCIAS_PARA_ARTIGO_2026-11-23.zip)
- [Manifesto das 6.300 imagens selecionadas](documentacao/manifesto_dataset_frutas6.json)
- [Seis prints das falhas iniciais](training/testes_externos/2026-11-23_rodada01/prints)
- [Fotos dos 11 testes iniciais](training/testes_externos/2026-11-23_rodada02_completa/imagens)
- [Log do treinamento](training/execucao.log)
- [Log da avaliação Fruits-360](training/avaliacao.log)
- [Log das verificações da interface e inferência](training/verificacao.log)

As seis falhas iniciais estão incluídas nos 11 arquivos; não somar as rodadas.
O manifesto registra arquivos e SHA-256, mas não inclui o próprio dataset.
Para conferir a consistência dos registros publicados, sem instalar TensorFlow:

```powershell
py -3.12 verificar_evidencias.py
```

Essa conferência verifica os artefatos já produzidos; não executa novo treinamento
nem substitui uma nova avaliação do modelo.
