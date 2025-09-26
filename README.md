# 🫀 Heart-Disease-Risk-Prediction

The Final Project of the GTC Machine Learning Internship that predicts heart disease risk using the UCI dataset. The pipeline covers preprocessing, modeling (supervised & unsupervised), evaluation, and a Streamlit web app with an ngrok tunnel for easy sharing.

---

#  ▶  Presentation Link   ◀
https://www.canva.com/design/DAG0AqCMFGE/YDVwuiIJjpIkyoXfbCeoGg/edit?utm_content=DAG0AqCMFGE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


## 🔍 Project phases
Short summary of what we did in the project:

- **Preprocessing & Cleaning & EDA** — missing values, type casts, distributions and initial visual exploration.  
- **Feature selection** — tested multiple methods: **Chi-square**, **RFE** (recursive feature elimination), and **Feature Importance** from a Random Forest.  
- **PCA analysis** — dimensionality reduction and visual inspection of principal components.  
- **Supervised learning** — trained classifiers to predict `target` (heart disease), evaluated with probability scores and metrics.  
- **Unsupervised learning** — clustering / exploratory methods to inspect structure in the data.  
- **Hyperparameter tuning** — grid/random search to obtain the final tuned pipeline.

Final model artifact: `final_tuned_pipeline.pkl` (used by the Streamlit app).

---

## 🚀 Quick install & run (Streamlit + ngrok)

> Replace values where needed (ngrok token, filenames). The examples below assume the repo name `Heart-Disease-Risk-Prediction` and the deploy script `deploy_with_pyngrok.py`.

1. **Clone the repo**
```bash
git clone https://github.com/Ali-Islam111/Heart-Disease-Risk-Prediction.git
cd Heart-Disease-Risk-Prediction
```
2. **Create & activate a virtual environment**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.bat
```

3. **Install requirements**
```bash
pip install -r requirements.txt
```

4. **Set your ngrok token (one of these options):**

Export as env var (recommended):

```bash
$env:NGROK_AUTHTOKEN="YOUR_NGROK_AUTHTOKEN"
```

Or open the deploy script and set `NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN"` (not ideal for public repos).

5. **Run using the deploy script (this starts Streamlit and opens ngrok):**
```bash
python deploy_with_pyngrok.py
```

## ⚠️ Important notes

Ngrok tunnels are just proxies — the public URL works only while the process (Streamlit + ngrok) is running somewhere. If you stop the process or shut the machine down, the link becomes unreachable.

To keep the app online 24/7, deploy to a server/VM and run the script as a background service, or use Streamlit Community Cloud / Render / Railway for a hosted solution.

Free ngrok URLs are ephemeral and change on each run; paid plans allow reserved subdomains.

## 📂 Useful files

`app.py` — Streamlit frontend.

`deploy_with_pyngrok.py` — helper to start Streamlit + open ngrok tunnel (one-command launch).

`final_tuned_pipeline.pkl` — trained model pipeline used by the app.

`02_heart_disease_preprocessed.csv` — preprocessed dataset used for EDA & PCA.

---
**Developed by:**
* Ali Islam Taha
* Ahmed Salah
* Mohamed Sameh
* Hassan Assar
* Mahmoud Khattab
* Mohamed Esam
