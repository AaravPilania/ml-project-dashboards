# Szeged Weather · Hidden Markov Model

Hourly Szeged weather 2006–2016 aggregated to days, labelled Sunny/Rainy, discretised into 18 observation symbols and modelled with a 2-state HMM (forward filter, Viterbi, 7-day forecast), following weather_prediction.ipynb.

## Run on Windows (PowerShell)

```powershell
cd szeged-weather-hmm
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. The dataset is bundled in `data/`, so no internet or Kaggle login is needed.

## Files
- `app.py`: the Streamlit dashboard (tabs: Overview · Explore · Model & metrics · Predict)
- `weather_model.py`: data prep and model code, reproduced from the group notebook
- `ui.py`: shared styling helpers (light theme, Plotly template)
- `data/weatherHistory.csv`: dataset (source: Kaggle budincsevity/szeged-weather)
