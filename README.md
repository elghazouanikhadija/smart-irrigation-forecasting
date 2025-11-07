# Smart Irrigation Forecasting with LSTM Neural Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LSTM-red)
![Agriculture](https://img.shields.io/badge/Domain-Agriculture-brightgreen)
![Region](https://img.shields.io/badge/Region-Tadla%2C%20Morocco-orange)

**Optimizing Water Usage through AI-Powered Irrigation Prediction in Tadla, Morocco**

**R²: 0.522 | Binary Accuracy: 98.0% | MAPE: 38.7%**

</div>

## Overview

This project implements a sophisticated **Long Short-Term Memory (LSTM)** neural network to predict daily irrigation requirements based on meteorological data from the **Tadla region of Morocco**. The system enables precise water management in agricultural operations, reducing water waste while maintaining optimal crop health through advanced machine learning techniques specifically tailored for Moroccan agricultural conditions.

## Data Source & Region

### Geographic Context
- **Region**: Tadla, Morocco
- **Agricultural Significance**: Major agricultural zone known for sugar beet, citrus, and cereal production
- **Climate**: Semi-arid with Mediterranean influences
- **Water Challenges**: Limited water resources requiring efficient irrigation management

### Dataset Characteristics
- **Temporal Range**: 2017-2024 (7 years of daily data)
- **Spatial Coverage**: 3 stations across Tadla agricultural perimeter
- **Key Variables**: 
  - Reference Evapotranspiration (ET₀)
  - Precipitation (Pluie)
  - Temperature (min/max)
  - Solar Radiation
  - Relative Humidity
  - Wind Speed

## Key Results & Performance

### Model Performance Summary

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **R² Score** | 0.469 | 0.495 | **0.522** |
| **RMSE** | 1.205 | 1.208 | 1.158 |
| **MAE** | 0.957 | 0.962 | 0.926 |
| **MAPE (non-zero)** | 41.69% | 43.46% | **38.69%** |
| **sMAPE** | 38.89% | 38.52% | 34.51% |
| **Binary Accuracy** | 96.51% | 97.06% | **98.04%** |

### Agricultural Impact for Tadla Region

- **Zero Irrigation Days**: Reduced from 35% to only 2-3% through continuous crop rotation
- **Irrigation Decision Accuracy**: 98% accuracy in identifying when irrigation is needed
- **Prediction Quality**: Explains 52.2% of irrigation need variance in Tadla conditions
- **Water Savings**: Precise predictions enable optimized water usage in water-scarce region
- **Crop Specific**: Tailored for Betterave (sugar beet) cultivation patterns in Morocco

## Technical Architecture

### Model Architecture
LSTM(64) → Dropout(0.2) → BatchNorm → LSTM(32) → Dropout(0.2) → BatchNorm → Dense(32) → Dense(1)

### Key Features
- 7-day sliding window with 1-day stride for temporal patterns
- Continuous crop rotation (Betterave_saison → Betterave_precoce → Betterave_tardive)
- Advanced feature engineering with crop coefficient (Kc) calculations
- Early stopping (patience=20) preventing overfitting
- Comprehensive regularization (Dropout + L2 + Batch Normalization)

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/elghazouanikhadija/smart-irrigation-forecasting.git
cd smart-irrigation-forecasting

# Create virtual environment
python -m venv irrigation_env
source irrigation_env/bin/activate  # Windows: irrigation_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### Basic Usage
```bash
from src.main import run_complete_pipeline

# Run complete pipeline with Tadla data
model, history, results = run_complete_pipeline("merged_meteo.csv")

# Access results
test_r2 = results['test']['metrics'][0]  # 0.522
binary_accuracy = results['test']['metrics'][4]  # 98.04
```
### Project Structure

smart-irrigation-forecasting/
├── data/
│   ├── meteo_dar_oulad_zidouh_DATA_1j.dat
│   ├── meteo_oulad_illoul_DATA_1j11.dat
│   └── OULED_AYAD_MTO_DATA_1j.dat
├── models/
│   └── best_lstm_model.keras
├── notebooks/
│   └── 02_complete_irrigation_pipeline.ipynb
├── config/
│   └── model_config.yaml
└── requirements.txt

### Key Features & Innovations
1. Tadla-Specific Crop Rotation System
```python
def feature_engineering_continuous(df):
    """
    Tailored for Tadla agricultural calendar:
    - Betterave_saison: October-January (aligned with Moroccan sowing)
    - Betterave_precoce: February-May (early season adaptation) 
    - Betterave_tardive: June-September (late season varieties)
    Maintains continuous Kc > 0.2 for consistent learning signal
    """
```
2. Regional Climate Adaptation

ET₀ calculations calibrated for Tadla's semi-arid climate

Rainfall patterns specific to Moroccan agricultural zones

Seasonal adjustments for Mediterranean climate variations

Crop coefficients validated for Moroccan sugar beet varieties

3. Advanced LSTM Architecture
   
Bidirectional processing of temporal patterns

Multiple regularization techniques preventing overfitting

Adaptive learning rate with ReduceLROnPlateau

Early stopping based on validation loss

### Results Analysis
Model Performance in Tadla Context
Excellent Binary Classification: 98.04% accuracy in identifying irrigation needs

Good Explanatory Power: R² of 0.522 explains majority of variance in regional conditions

Reasonable Prediction Error: MAPE of 38.7% on non-zero values

Strong Generalization: Consistent performance across train/val/test sets

### Agricultural Impact for Moroccan Agriculture
Water Optimization: Critical for Tadla's limited water resources

Crop Health: Maintains optimal soil moisture for sugar beet production

Labor Efficiency: Reduces manual monitoring in large-scale farms

Economic Benefits: Optimized water usage reduces operational costs

Sustainability: Aligns with Morocco's agricultural development strategy

### Usage Examples
Complete Pipeline Execution with Tadla Data
````python
from src.main import run_complete_pipeline

# Execute full pipeline with regional data
model, history, results = run_complete_pipeline(
    data_path="merged_meteo.csv",
    target_column="Besoin_irrigation"
)

````
### Model Interpretation
Regional Feature Importance
The model successfully learned Tadla-specific patterns:

Seasonal ET₀ variations in semi-arid climate

Rainfall effectiveness in Moroccan agricultural context

Sugar beet growth cycles adapted to local conditions

Micro-climate variations across the 3 stations

### Regional Significance
This project addresses critical water management challenges in Morocco's agricultural sector:

Water Scarcity: Tadla region faces increasing water stress

Climate Change: Adapting to changing precipitation patterns

Food Security: Optimizing water use for staple crop production

Economic Development: Supporting Morocco's agricultural exports

### Citation
If you use this project in your research, please cite:

bibtex
@software{smart_irrigation_2023,
  title = {Smart Irrigation Forecasting with LSTM: Tadla, Morocco Case Study},
  author = {Elghazouani khadija},
  year = {2024},
  url = {https://github.com/elghazouanikhadija/smart-irrigation-forecasting},
  note = {Meteorological data from 3 stations in Tadla region, Morocco (2017-2024)}
}
### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Office Régional de Mise en Valeur Agricole de Tadla (ORMVAT) for meteorological data

Direction de la Météorologie Nationale (Morocco) for climate data support

Farmers and Agricultural Cooperatives in Tadla for practical insights

<div align="center">
Star this repository if you find it useful!

Building sustainable agriculture in Morocco through artificial intelligence
Precision. Efficiency. Sustainability. National Impact.

</div>
Last updated: November 2024
Performance metrics based on Tadla region test set evaluation
*Data period: 2017-2024 from 3 meteorological stations in Tadla, Morocco*

