# Dashboard de enfermedad cardiaca

App web en Python con Dash, Flask y Plotly basada en `heart_disease_clean.csv`.

## Ejecutar

```powershell
cd C:\Users\admin\Documents\Codex\2026-06-28\qui\outputs\heart_dash_app
pip install -r requirements.txt
python app.py
```

Luego abre:

```text
http://127.0.0.1:8050
```

## Que incluye

- Mas de 4 graficas: distribucion de edad, boxplot numerico, barras categoricas, heatmap de correlacion y analisis de normalidad por asimetria/curtosis.
- Controladores interactivos: rango de edad, objetivo binario/severidad, variable numerica, variable categorica, numero de arboles, tamano de prueba y profundidad del modelo.
- Random Forest integrado con exactitud, F1 macro, matriz de confusion, importancia de variables y prediccion individual.

## Sugerencia de modelado

Para una primera version conviene usar `num > 0` como clasificacion binaria, porque las clases 2, 3 y 4 tienen menos muestras. La app permite visualizar la severidad original, pero el predictor individual usa el enfoque binario para mayor estabilidad.
