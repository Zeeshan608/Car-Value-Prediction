<div align="center">

# 🚘 AutoWorth

### AI-Powered Used Car Price Estimator

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-7C5CFF)
![Plotly](https://img.shields.io/badge/Charts-Plotly-3F4F75?logo=plotly&logoColor=white)

*Describe a vehicle. Get an instant, data-driven market price.*

</div>

---

## ✨ Features

- ⚡ **Instant valuation** from a trained LightGBM regression model
- 🎯 **Price range** with an interactive gauge chart
- 📊 **Feature importance** chart showing what drives the price
- 🎨 **Modern dark UI** with gradient cards and responsive layout
- 📂 **Optional dataset upload** for exact model-level popularity values

## 🗂️ Project Structure

```
car_price_app/
├── app.py                          # Streamlit frontend
├── lightgbm_car_price_model.pkl    # Trained model
├── requirements.txt                # Dependencies
└── .streamlit/config.toml          # Dark theme
```

## 🚀 Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 🧠 Model Overview

| Item | Detail |
|------|--------|
| Algorithm | LightGBM (gradient boosting) |
| Target | `log(price)`, converted back to USD |
| Features | Brand, fuel type, transmission, exterior/interior colour, accident history, clean title, mileage, car age, horsepower, engine size, model popularity |

> **Note:** Model popularity (`model_freq`) is the share of listings for a car model in the training data. Select an approximate level in the app, or upload the original `used_cars.csv` in the sidebar for exact values.

## 🛠️ Troubleshooting

**`Could not find module ... lib_lightgbm.dll` (Windows)**

1. Install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and restart your PC.
2. Reinstall LightGBM:
   ```bash
   pip uninstall lightgbm -y
   pip install --no-cache-dir lightgbm
   ```
3. Confirm you're on 64-bit Python:
   ```bash
   python -c "import struct; print(struct.calcsize('P')*8)"
   ```

**Unpickling error:** install the same LightGBM version used for training (`pip show lightgbm` in your training environment).

## ⚠️ Disclaimer

Estimates are indicative only and not a formal vehicle valuation.

---

<div align="center">

Built with ❤️ using **Streamlit · LightGBM · Plotly**

</div>
