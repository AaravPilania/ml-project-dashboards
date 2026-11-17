# ML Project Dashboards

Three standalone Streamlit dashboards, one per folder:

- `customer-segmentation/`: K-Means customer segmentation
- `szeged-weather-hmm/`: Szeged weather prediction with a Hidden Markov Model
- `breast-cancer-bagging/`: Breast Cancer Wisconsin with bagged decision trees

Run any one (Windows PowerShell):

```
cd customer-segmentation
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
