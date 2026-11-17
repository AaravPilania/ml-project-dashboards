# Customer Segmentation (K-Means)

K-Means (k = 4) on 2,627 customers, following customer_segmentation.ipynb: median/mode imputation, ordinal Spending_Score, one-hot encoding, StandardScaler, K-Means (random_state 42, n_init 10).

## Run on Windows (PowerShell)

```powershell
cd customer-segmentation
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. The dataset is bundled in `data/`, so no internet or Kaggle login is needed.

## Files
- `app.py`: the Streamlit dashboard (tabs: Overview · Explore · Model & metrics · Predict)
- `seg_model.py`: data prep and model code, reproduced from the group notebook
- `ui.py`: shared styling helpers (light theme, Plotly template)
- `data/Test.csv`: dataset (source: Kaggle abisheksudarshan/customer-segmentation, Test.csv (the file the notebook reads))
