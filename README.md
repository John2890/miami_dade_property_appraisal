# 🏠 Miami-Dade Real Estate Automated Valuation Model (AVM)

An end-to-end machine learning pipeline and interactive web application that predicts residential property sale prices in Miami-Dade County. 

Unlike standard academic exercises (e.g., the Ames Housing dataset), this project processes, cleans, and merges **three distinct, real-world government datasets** covering nearly one million local property records to train a Random Forest model on ~45,000 qualified residential sales transactions.

**[🚀 Live Interactive App →][(https://your-app-url.streamlit.app)** *(replace with your Streamlit URL)*](https://miamidadepropertyappraisal-kyf7bqeregtyljqfdx3o7n.streamlit.app)

---

## 📌 Project Overview
The objective of this project was to engineer a robust Automated Valuation Model (AVM) that accurately reflects the highly nuanced, localized dynamics of the South Florida real estate market. It encompasses the full data lifecycle: from raw municipal data extraction and feature engineering to model training and full-stack web deployment.

---

## 🗄️ Data Architecture

To build a comprehensive feature matrix, three disparate datasets were merged using the unique 12-digit `PARCEL_ID` / `FOLIO` identifier, achieving a 99.8% match rate.

| Source | Description | Contribution to Model |
|---|---|---|
| **Florida DOR NAL File** | State tax roll database | Square footage, year built, effective year built, assessed values, property use codes. |
| **Florida DOR SDF File** | State sales database | Actual transaction prices, sale dates, qualification codes, multi-parcel flags. |
| **Miami-Dade Open Data Hub**| County Property Point View | Bedroom count, bathroom count, floor count (metrics absent from state-level reporting). |

---

## ⚙️ Methodology

### 1. Data Filtering & Sanitization
Raw sales data was aggressively filtered to isolate qualified, arm's-length transactions:
- `QUAL_CD == 1`: Restricted to fair-market transactions between unrelated parties.
- **Single-Parcel Sales:** Excluded bulk developer deals (`MULTI_PAR_SAL` is null).
- **Price Boundaries:** Excluded symbolic transfers (< $10,000) and extreme luxury outliers (> $2,000,000). Removing the ultra-luxury segment (~5% of sales) drastically reduced model error for the standard residential market.
- **Zoning:** Restricted to Use Codes `1` (Single Family) and `4` (Condominium).

### 2. Feature Engineering
Raw data fields were transformed into behavioral and structural metrics that better align with real estate appraisal standards:

* **`eff_age` (Continuous):** Calculated as `Current Year - Effective Year Built`. This accounts for renovations, allowing the model to recognize that a gutted and flipped 1950s home operates functionally closer to a new build.
* **`age_tier` (Categorical):** Binned original build years into New (<10 yrs), Modern (10–40), Mid-Century (40–75), and Historic (75+). 
* **`sqft_per_bedroom`:** Measures architectural layout and spaciousness.
* **`bath_bed_ratio`:** Acts as a proxy for construction finish level and modernization.
* **`zip_dummies`:** One-hot encoded the top 14 zip codes by sales volume to capture neighborhood-level geographic premiums without over-sparsifying the matrix.

*Note: Assessed county value (`just_value`) was deliberately excluded to prevent data leakage, as it correlates at 0.98 with the target variable.*

### 3. Model Selection & Performance
A **Random Forest Regressor** was selected over Linear Regression to better capture non-linear relationships (e.g., the compounding value of specific zip codes mixed with modern renovations).

* **Algorithm:** Random Forest Regressor (`n_estimators=100`, `max_depth=20`, `min_samples_leaf=3`)
* **R² Score:** ~0.77
* **Mean Absolute Error (MAE):** ~$100,000 (~18% of median sale price)
* **Training Volume:** ~45,000 records

---

## 🧠 Key Engineering Challenges & Solutions

**Overcoming Data Sparsity & Feature Bias (The "Historic Premium")**
During initial testing, the model exhibited a severe hallucination: it overvalued unrenovated 1940s homes located in newer, less affluent suburbs. 

* **The Root Cause:** The model learned a massive "Historic Premium" because the vast majority of pre-1950 homes in the training data survived in hyper-affluent areas (e.g., Coral Gables, Miami Beach). When presented with a 1940s home in a standard suburb, the model lacked local historical data and erroneously applied the global historic premium.
* **The Solution:** The logic was refactored to decouple the physical age from the functional age. By converting `year_built` into categorical bins (`age_tier`) and keeping `eff_age` (renovation age) continuous, the model was forced to evaluate older properties based on their modernization status and geographic coordinates rather than raw age alone. 

---

## 💻 Tech Stack & Deployment

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
- **Version Control:** Git / GitHub

---

## 👨‍💻 About the Developer

Built by **Jonathan Rodriguez** — Electrical Engineering student at Florida International University (FIU) with concentrations in Power Systems and Data Systems Software. 

This project was developed to demonstrate end-to-end data pipeline architecture, machine learning implementation, and product deployment. 

*(Training notebook available within repository).*
