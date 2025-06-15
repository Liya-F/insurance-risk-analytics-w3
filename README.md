# Insurance Risk Analytics

A modular and reproducible pipeline for analyzing insurance risk data with an emphasis on trends, financial insights, outlier detection, and geographic-temporal patterns. This project leverages exploratory data analysis (EDA), visualization, and version-controlled datasets to support actionable insights for Alpha Insurance Company’s strategy.

---

## Key Features

- Modular code for reusable EDA and visualization components
- Insightful univariate, bivariate, and multivariate analyses
- Loss ratio calculations across Province, Gender, and Vehicle Type
- Outlier detection using statistical plots
- DVC integration for data versioning and reproducibility
- Creative and insightful visualizations for business storytelling

---
## Analysis Pipeline

### 1. Data Cleaning
- Drop columns with all missing values
- Handle missing rows and inconsistent formats
- Standardize column names and date formats
- Replace the main dataset with cleaned data

### 2. Exploratory Data Analysis
- **Univariate**: Distributions of key variables (e.g., `TotalPremium`, `TotalClaims`)
- **Bivariate/Multivariate**: Relationships between `TotalPremium`, `TotalClaims`, and `ZipCode`, temporal trends, and loss ratios
- **Outliers**: Box plots for numerical columns to detect skewness or anomalies

### 3. Visualization
Three creative insights visualized:
- Loss Ratio variation across `Province`, `Gender`, and `VehicleType`
- Claim severity and frequency changes over 18 months
- Top Vehicle Makes associated with highest and lowest claims

### 4. Data Version Control (DVC)
- `dvc init` and remote setup with `dvc_storage/`
- Dataset tracking using `dvc add`
- Commit `.dvc` files and push datasets to local remote
- Ensure data integrity and reproducibility

---
## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/insurance-risk-analytics-w3.git
cd insurance-risk-analytics-w3 
```
### 2. Set Up a Virtual Environment
```bash 
python -m venv .venv
```
#### Activate the environment:
```bash 
.venv\Scripts\activate #On Windows
```
```bash 
source .venv/bin/activate #On macOS/Linux:
```
#### Install Python Dependencies
```bash 
pip install -r requirements.txt
```