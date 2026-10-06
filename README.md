# Household Energy Forecasting with an Agentic Energy Advisor

Forecasts hourly household electricity demand (UCI dataset) and uses the forecast to schedule shiftable appliances away from peaks.

## Results (test set, Apr-Nov 2010)
| Model | 1h MAE | 24h MAE | 168h MAE |
|---|---|---|---|
| Naive yesterday | 0.505 | 0.505 | 0.527 |
| Linear Regression | 0.346 | 0.439 | 0.455 |
| Random Forest | 0.301 | 0.419 | 0.438 |
| XGBoost | **0.298** | **0.413** | **0.427** |
| LSTM | 0.379 | 0.424 | n/a |
| Transformer | 0.341 | 0.422 | n/a |

Agent: 13.3% peak reduction with forecasts vs 63.1% with a perfect-forecast oracle.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Download "Individual household electric power consumption" from the UCI repository and put `household_power_consumption.txt` in `data/raw/`.

## Run order
```
$env:PYTHONPATH = "src"
python src/data_prep.py
python src/eda.py
python src/features.py
python src/baselines_trees.py
python src/lstm.py
python src/transformer.py
python src/compare.py
python src/agent.py
python src/prep_dashboard.py
python -m streamlit run app/dashboard.py
```