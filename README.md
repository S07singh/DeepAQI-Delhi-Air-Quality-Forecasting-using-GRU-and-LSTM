# 🌫️ DeepAQI — Delhi Air Quality Forecasting using GRU & LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S07singh/DeepAQI-Delhi-Air-Quality-Forecasting-using-GRU-and-LSTM/blob/main/Full_Data_GRU_LSTM_Training.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready deep learning project that forecasts Delhi's **Air Quality Index (AQI)** using **GRU** and **LSTM** recurrent neural networks, trained on hourly multivariate data from **14 monitoring stations** across Delhi (2000–2024).

---

## 🚀 Features

- **Full 14-station aggregation** — City-wide hourly average across all Delhi monitoring stations for a clean, representative AQI signal
- **Comprehensive EDA** — Distribution plots, correlation heatmaps, seasonal decomposition, and temporal trend analysis
- **Feature engineering** — Temporal features (hour, day, month), lag features (1h, 24h), and rolling statistics (24h mean & std)
- **Deep architectures** — 3-layer stacked GRU and LSTM with L2 regularization, dropout, and gradient clipping
- **Hyperparameter tuning** — Random search over architecture size, dropout rate, learning rate, and batch size
- **Chronological split** — 70/15/15 train/val/test to prevent data leakage
- **Model comparison** — MSE, MAE, RMSE, R², training time, and loss curve analysis
- **Streamlit web app** — Real-time next-hour AQI prediction with category indicator

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [HuggingFace — abhinavsarkar/delhi_air_quality_feature_store_processed.csv](https://huggingface.co/datasets/abhinavsarkar/delhi_air_quality_feature_store_processed.csv) |
| **Raw Size** | 2,921,413 rows × 15 columns |
| **Stations** | 14 monitoring stations across Delhi |
| **Time Range** | 2000 – 2024 (hourly) |
| **After Aggregation** | ~208,000 hourly city-wide averages |

### Columns

| Category | Features |
|----------|----------|
| **Weather** | `temperature`, `humidity`, `pressure`, `wind_speed`, `wind_direction` |
| **Pollutants** | `pm25`, `pm10`, `no2`, `so2`, `o3`, `co` |
| **Target** | `aqi` (Air Quality Index) |
| **Identifiers** | `location_id`, `city`, `event_timestamp` |

The raw dataset contains readings from 14 monitoring stations. During preprocessing, all stations are **aggregated (averaged)** per hourly timestamp to create a single city-wide time series, which is more robust and representative than any individual station.

---

## 📁 Project Structure

```
DeepAQI/
├── Full_Data_GRU_LSTM_Training.ipynb   # Training notebook (Colab + GPU)
├── app.py                              # Streamlit prediction app
├── data.ipynb                          # Dataset loading & parquet conversion
├── model/
│   ├── gru_model.keras                 # Trained GRU weights
│   ├── lstm_model.keras                # Trained LSTM weights
│   ├── scaler.pkl                      # Fitted MinMaxScaler (X and y)
│   └── metadata.pkl                    # Features, seq_len, hyperparameters
├── data/
│   └── delhi_pollution_hourly.parquet  # Full 14-station dataset
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🏗️ Model Architecture

Both models use a **3-layer stacked architecture** with strong regularization to balance capacity and generalization:

| Component | Configuration |
|-----------|---------------|
| **Layer 1** | GRU/LSTM (64 units) + Dropout |
| **Layer 2** | GRU/LSTM (32 units) + Dropout |
| **Layer 3** | GRU/LSTM (16 units) + Dropout |
| **Dense** | 16 (ReLU) → 1 (Linear) |
| **Regularization** | L2 (1e-3) on kernel + recurrent weights |
| **Optimizer** | Adam with gradient clipping (clipnorm=1.0) |
| **Loss** | MSE |
| **Callbacks** | EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5) |

### Results

| Model | MSE | MAE | RMSE | R² |
|-------|-----|-----|------|-----|
| **GRU** | 218.02 | 13.54 | 14.77 | **0.3244** |
| LSTM | 553.27 | 18.98 | 23.52 | -0.71 |

> **GRU is the recommended model** with an R² of 0.32, meaning it explains ~32% of AQI variance — reasonable for a complex environmental time series.

---

## ⚡ Quick Start

### 1. Training (Google Colab)

Click the **Open in Colab** badge above, or:

1. Open `Full_Data_GRU_LSTM_Training.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Upload `delhi_pollution_hourly.parquet` when prompted
4. **Run all cells** — the notebook will:
   - Aggregate 14 stations into hourly city-wide averages
   - Perform EDA and feature engineering
   - Run hyperparameter search (6 trials each for GRU & LSTM)
   - Train final models and evaluate on the test set
   - Trigger automatic download of model files
5. Place downloaded files into the `model/` directory

### 2. Streamlit App (Local)

```bash
# Clone the repo
git clone https://github.com/S07singh/DeepAQI-Delhi-Air-Quality-Forecasting-using-GRU-and-LSTM.git
cd DeepAQI-Delhi-Air-Quality-Forecasting-using-GRU-and-LSTM

# Create virtual environment
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Upload a CSV or Parquet file with recent hourly air quality readings — the app will preprocess the data and predict the **next-hour AQI** using the selected model.

---

## 🎯 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0 – 50 | 🟢 Good | Minimal |
| 51 – 100 | 🟢 Satisfactory | Minor breathing discomfort to sensitive people |
| 101 – 200 | 🟡 Moderate | Breathing discomfort to people with lung/heart disease |
| 201 – 300 | 🟠 Poor | Breathing discomfort on prolonged exposure |
| 301 – 400 | 🔴 Very Poor | Respiratory illness on prolonged exposure |
| 401 – 500 | 🟣 Severe | Affects healthy people, serious impact on those with existing diseases |

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Time Series Analysis**: Statsmodels
- **Web App**: Streamlit
- **Dataset Source**: HuggingFace Datasets

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
