# Modelo treinado

`treinar_modelo.py` gera `fruit_model.keras`, `classes.json` e `training.json`.
A normalização está dentro da rede (`preprocessing: none`). O campo
`resize: bilinear_stretch` exige o redimensionamento compartilhado em
`utils/model_data.py`. A ordem de `classes` corresponde às saídas.
O aplicativo rejeita metadados incompatíveis e não usa o modelo genérico.
