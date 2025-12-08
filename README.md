# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview
This project delivers a production-ready insurance risk analytics solution for **AlphaCare Insurance Solutions (ACIS)**. Using historical car insurance data in South Africa, the system identifies low-risk segments, optimizes premiums, and enables data-driven decision-making for underwriting and marketing teams.

The solution encompasses:

- Data ingestion and versioned storage
- Exploratory data analysis for risk insights
- Hypothesis-driven statistical validation
- Predictive modeling for claim severity and premium optimization
- Reproducible and auditable pipelines with DVC

---

## Business Objective
ACIS aims to:

- Attract low-risk customers with competitive premiums  
- Adjust pricing strategies based on data-driven insights  
- Ensure regulatory compliance through reproducible and auditable pipelines  

Key business questions addressed:

1. Which regions, vehicle types, and customer demographics are associated with lower risk?  
2. How can premiums be optimized based on predicted claim probability and severity?  
3. How can the analytics workflow be version-controlled and reproducible for audits?

---

## Deliverables & Approach

### Task 1: Data Exploration & Analysis
- Comprehensive data quality assessment: missing values, distributions, and outliers  
- Temporal and geographic trend analysis of claims and premiums  
- Visualizations highlighting high- and low-risk segments  
- Delivered actionable insights to guide segmentation and premium strategy  

### Task 2: Reproducible Data Pipeline
- Initialized **DVC** to version datasets and pipelines  
- Configured local remote storage to store and track raw and processed data  
- All datasets are DVC-tracked; code and pipeline changes versioned in Git  
- Enables reproducibility for future updates, audits, and model retraining  

### Task 3: Statistical Validation
- Defined risk metrics: Claim Frequency, Claim Severity, and Profit Margin  
- Designed and executed A/B tests to validate risk differences across:  
  - Provinces and zip codes  
  - Gender and customer attributes  
  - Profit margins  
- Generated business-ready insights to inform pricing and marketing decisions  

### Task 4: Predictive Modeling
- **Claim Severity Model:** Regression to predict total claim amounts  
- **Premium Optimization Model:** Risk-based premium estimation incorporating probability of claim and severity  
- Algorithms: Linear Regression, Random Forest, XGBoost  
- Interpretability: SHAP used to identify top 5–10 risk drivers and quantify impact on premiums  
- Delivered robust, explainable models ready for deployment

---

## Technology Stack
- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Machine Learning:** XGBoost, Random Forest, Linear Regression  
- **Data Versioning:** DVC for data and pipeline management  
- **Version Control & CI/CD:** Git, GitHub Actions  
- **Visualization & Reporting:** Jupyter Notebooks, automated plots  

---

