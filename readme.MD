# 📈 Stock Price Prediction Platform

This repository contains a dual-interface system for stock price prediction:

- A **Streamlit-based web app** (`final_model.py`) for visualizing and predicting stock prices.
- A **FastAPI backend** (`main.py`) for accessing prediction APIs and integrating services.
- A **comparison module** (`compare.py`) to evaluate predicted vs. actual prices.
- A CI/CD workflow setup using GitHub Actions.

## 📁 Folder Structure

. ├── backend/ │ ├── compare_utils.py │ ├── final_models_utils.py │ └── ... ├── .github/workflows/ │ └── dev-branch-checks.yml ├── final_model.py # Streamlit frontend ├── main.py # FastAPI backend ├── compare.py # Prediction comparison logic ├── rf.py # Random Forest model training ├── lstm_model.py # LSTM sentiment model ├── lstm.h5 # LSTM model weights ├── lstm.txt # LSTM summary/info ├── random_forest.txt # Random Forest summary/info ├── scaler1.pkl # Feature scaler ├── sentiment_scaler1.pkl # Sentiment score scaler ├── target_scaler1.pkl # Target scaler ├── tokenizer1.pkl # Tokenizer for sentiment data ├── requirements.txt ├── setup.cfg └── README.md

---

## ⚙️ Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/AkankshyaKar106/Stock_Prediction
cd Stock_Prediction

2. Install dependencies

pip install -r requirements.txt

3. Ensure the following model files are present in the root directory:

- lstm.h5
- tokenizer1.pkl
- scaler1.pkl
- sentiment_scaler1.pkl
- target_scaler1.pkl

4. Run the Streamlit App (Frontend)

- Launch the interactive web interface for predictions:
streamlit run final_model.py

- Launch the interactive web interface for predictions:
streamlit run compare.py

5. Run the FastAPI Server (Backend)

Start the API server:
uvicorn main:app --reload --port 8002
Once running, visit http://127.0.0.1:8002/docs to explore and test API endpoints via Swagger UI.

6. ✅ GitHub CI/CD

CI/CD is managed using GitHub Actions. The workflow file:
.github/workflows/dev-branch-checks.yml
