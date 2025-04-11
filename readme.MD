# 📈 Stock Price Prediction Platform

This repository provides a full-stack stock prediction system using:

- 🔹 **Streamlit App** (`final_model.py`): A web interface for predicting and visualizing stock prices.
- 🔹 **FastAPI Backend** (`main.py`): Provides prediction APIs and integration endpoints.
- 🔹 **Comparison Module** (`compare.py`): Compares predicted vs. actual prices.
- 🔹 **CI/CD** via GitHub Actions for streamlined development workflows.

---

## 📁 Folder Structure

```plaintext
.
├── backend/
│   ├── compare_utils.py             # Utilities for compare.py
│   ├── final_models_utils.py        # Utilities for final_model.py
│
├── .github/
│   └── workflows/
│       └── dev-branch-checks.yml    # CI/CD workflow
│
├── final_model.py                   # Streamlit frontend
├── main.py                          # FastAPI backend
├── compare.py                       # Prediction vs actual analysis
├── rf.py                            # Random Forest training
├── lstm_model.py                    # LSTM-based sentiment model
│
├── lstm.h5                          # Trained LSTM model weights
├── lstm.txt                         # LSTM summary
├── random_forest.txt                # Random Forest model summary
├── tokenizer1.pkl                   # Tokenizer for sentiment input
├── scaler1.pkl                      # Feature scaler
├── sentiment_scaler1.pkl            # Sentiment score scaler
├── target_scaler1.pkl               # Target variable scaler
│
├── requirements.txt                 # Python dependencies
├── setup.cfg
└── README.md
```
---

## ⚙️ Setup Instructions
### 1. 🔽 Clone the Repository
```bash 
git clone https://github.com/AkankshyaKar106/Stock_Prediction
```
```bash 
cd Stock_Prediction
```

### 2. 📦 Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3. 📂 Ensure the Following Model Files Are Present in the Root Directory
- lstm.h5  
- tokenizer1.pkl  
- scaler1.pkl  
- sentiment_scaler1.pkl  
- target_scaler1.pkl

### 4. 🚀 Run the Streamlit App (Frontend)
Prediction App:
```bash 
streamlit run final_model.py
```

Comparison App:
```bash 
streamlit run compare.py
```

### 🔗 Access the App  
[Streamlit App](http://localhost:8501)

### 5. ⚡ Run the FastAPI Server (Backend)
```bash 
uvicorn main:app --reload --port 8002
```

Once running, open the API docs at:  
[👉 FastAPI Docs](http://127.0.0.1:8002/docs)

### 6. ✅ GitHub CI/CD  
CI/CD is managed using GitHub Actions. The workflow file is located at:  
[🔧 GitHub Actions Workflow](.github/workflows/dev-branch-checks.yml)
