# 🏥 Health Insurance Cost Predictor

A machine learning solution that predicts health insurance premiums based on demographic, financial, and health-related inputs. The project uses age-based model segmentation for improved accuracy and is deployed with a user-friendly Streamlit interface.

---

## 🚀 Key Highlights

- **Age-Based Dual-Model Architecture**  
  - **Under 26**: Linear Regression with enhanced genetical risk features  
  - **26+ Age Group**: XGBoost with standardized input features  
- **High Accuracy**  
  - R² > 0.98 across both segments  
- **Interactive Web App**  
  - Built with Streamlit for real-time predictions  
- **Feature-Rich Input**  
  - Demographics, income, BMI, genetic risks, smoking, and plan type  
- **Clean Codebase**  
  - Modular structure, model artifacts, and automated preprocessing

---

## 🧠 Methodology

1. **Data Preprocessing**  
   - Cleaning, feature engineering, outlier handling, scaling  
2. **Model Development**  
   - Cross-validation, hyperparameter tuning  
3. **Segmentation Strategy**  
   - Age-based modeling due to error pattern analysis  
4. **Deployment Pipeline**  
   - Age-aware model selection and prediction via Streamlit

---

## 📈 Performance Summary

| Group     | Model            | RMSE    | R²     |
|-----------|------------------|---------|--------|
| **<= 25** | Linear Regression| 292.80  | 0.9887 |
| **> 25**  | XGBoost          | 373.55  | 0.9971 |

- Unified preprocessing ensures smooth model switching  
- Genetical risk data boosts prediction for younger demographics

---

## 🌐 Live Demo

🔗 **Try it here**: [https://health-insurance-premium.streamlit.app/](https://health-insurance-premium.streamlit.app/)

An interactive web interface where users can enter their information and instantly get a personalized health insurance cost prediction based on their age group.

**Input Categories**:
- Age, Gender, Income, Employment, BMI, Marital Status, Dependents, Medical History, Smoking, Genetic Risk, Insurance Plan, Region

---

## 📁 Project Structure in Detail

- **models/**: Contains all notebooks, artifacts, and datasets used in model training and evaluation
  - **artifacts/**: Serialized models and scalers used for prediction  
    - **model_rest.joblib**: XGBoost model for age 25+ group  
    - **model_young.joblib**: Linear Regression model for young age group  
    - **scaler_rest.joblib**: Scaler used for features in the 25+ age group  
    - **scaler_young.joblib**: Scaler used for features in the young age group  
  - **data_segmentation.ipynb**: Notebook for segmenting data by age group  
  - **ml_premium_prediction.ipynb**: Initial model training notebook  
  - **ml_premium_prediction_rest.ipynb**: Model training for age 25+ group after data segmentation
  - **ml_premium_prediction_rest_with_gr.ipynb**: Model training with dummy genetic risk for 25+ group  
  - **ml_premium_prediction_young.ipynb**: Model training for the young group after data segmentation
  - **ml_premium_prediction_young_with_gr.ipynb**: Young group model training with genetic risk  
  - **premiums.xlsx**: Original dataset with all records  
  - **premiums_rest.xlsx**: Filtered dataset for age 25+ group  
  - **premiums_young.xlsx**: Filtered dataset for young age group  
  - **premiums_young_with_gr.xlsx**: Young group data including genetic risk info  

- **main.py**: Entry point for the Streamlit app – handles user input and prediction logic  
- **prediction_helper.py**: Utility functions for model loading and feature preprocessing  
- **requirements.txt**: Lists all required Python libraries and versions  
- **README.md**: Project overview, setup instructions, methodology, and contribution guidelines  

---


## 🛠️ Setup Instructions

### 🔧 Run Locally

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Arka-Dutta-28/Health_Insurance_Premium_Prediction.git
   cd Health_Insurance_Premium_Prediction
   
2. **Create a virtual environment (optional but recommended)**   
   ```commandline
    python -m venv venv
    source venv/bin/activate
   ```

3. **Run the Streamlit app:** 
   ```commandline
    streamlit run frontend/main.py
   ```
---

## 💻 Tech Stack

- **ML**: Scikit-learn, XGBoost  
- **Data Handling**: Pandas, NumPy  
- **Web App**: Streamlit  
- **Visualization**: Matplotlib, Seaborn  
- **Model Persistence**: Joblib  

---

## 🙋‍♂️ Usage

1. Launch the app 
2. Enter your personal and health details  
3. Click **Predict**  
4. The model appropriate for your age group gives a real-time estimate  

---

## 🤝 Contributing

Contributions welcome!  
Open an issue to discuss changes or submit a pull request.

---

## 📜 License

Licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 📬 Contact

Email: **dutta280302@gmail.com**  
> *This project is for educational use. Consult a professional for real insurance advice.*
