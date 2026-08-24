# Notebooks

## `00_recorrido_completo.ipynb`

Recorre todo el proyecto paso a paso, con los números a la vista. Es lo que hay que abrir
para entender qué hicimos y por qué.

```powershell
jupyter lab notebooks/00_recorrido_completo.ipynb
```

**Antes hay que tener el entorno y los datos:**

```powershell
.\scripts\setup_env.ps1
python -m ingestion.run
python -m transform.silver
python -m features.gold_tp
```

El notebook no reimplementa nada: cada paso llama a los módulos del repo, así que lo que
corre ahí es exactamente lo que corre en producción. Si un número del notebook cambia, es
porque cambió el pipeline.

Tarda unos minutos: la sección 10 hace 38 reentrenamientos (el walk-forward).

---

## Por qué el `.ipynb` se genera desde un `.py`

El notebook **se produce con `python notebooks/00_recorrido_completo.py`**, no se edita a
mano. Un `.ipynb` es JSON con el código embebido línea por línea: editado directamente,
los diffs de git son ilegibles y los merges entre dos personas son un infierno.

Generándolo desde un `.py`:

- el contenido se versiona como texto legible y los diffs se entienden;
- no se commitean salidas de ejecución, que inflan el repo y cambian en cada corrida;
- los `id` de celda se derivan del contenido con un hash, así que **regenerar sin cambios
  produce un archivo byte a byte idéntico** — si fueran aleatorios, cada corrida ensuciaría
  el diff.

Para cambiar el notebook: se edita el `.py` y se regenera.
