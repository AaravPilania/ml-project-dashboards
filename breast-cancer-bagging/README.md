# Breast Cancer Wisconsin · Bagged Decision Trees

Single decision tree vs bagging of 100 trees (OOB score, 25-fold repeated CV, ROC, feature importance), following bagging_decision_tree_breast_cancer.ipynb.

## Run on Windows (PowerShell)

```powershell
cd breast-cancer-bagging
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. The dataset is bundled in `data/`, so no internet or Kaggle login is needed.

## Files
- `app.py`: the Streamlit dashboard (tabs: Overview · Explore · Model & metrics · Predict)
- `cancer_model.py`: data prep and model code, reproduced from the group notebook
- `ui.py`: shared styling helpers (light theme, Plotly template)
- `data/data.csv`: dataset (source: Kaggle uciml/breast-cancer-wisconsin-data)
