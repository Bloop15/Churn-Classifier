# Telco Customer Churn Prediction

A machine learning project to predict customer churn, a binary classification problem, using the Telco Customer Churn dataset. The project follows a complete, end-to-end data science workflow, from **EDA and preprocessing** to **model training and evaluation**. The goal is to identify the most effective classifier model and uncover the key factors driving customer attrition.

---

## Project Workflow

This project follows a structured, modular workflow, documented across four Jupyter Notebooks:

- **01_data_inspection.ipynb**: Initial data loading, cleaning, and comprehensive Exploratory Data Analysis (EDA) to uncover key insights into the dataset's characteristics and relationships.  
- **02_data_preprocessing.ipynb**: Prepares the data for modeling by handling feature scaling, categorical encoding, and splitting the dataset into training and testing sets.  
- **03_model_training.ipynb**: Trains and evaluates multiple classifiers, including a baseline model, Logistic Regression, Random Forest, and XGBoost. Includes hyperparameter tuning to optimize performance.  
- **04_model_visualization_and_analysis.ipynb**: Creates key visualizations, analyzes feature importance, and presents final conclusions.  

---

## Key Findings from EDA

- **Class Imbalance**:  
  The dataset is imbalanced, with ~73% customers not churning. This was handled using stratified data splitting and `class_weight='balanced'` in our models.  

- **Strong Predictors of Churn**:  
  - **Contract Type**: Customers on month-to-month contracts churn at a much higher rate.  
  - **Tenure**: Customers with shorter tenure are significantly more likely to churn.  
  - **Monthly Charges**: Higher monthly charges are associated with higher churn.  

---

## Final Model Performance

The best models were fine-tuned and evaluated on a hold-out test set. The **F1-score for the churn class** was the primary metric for comparison.

| Model                        | Accuracy | F1-Score (Churn) | ROC-AUC |
|------------------------------|----------|------------------|---------|
| Random Forest                | 0.7527   | 0.6266           | 0.8352  |
| XGBoost                      | 0.7392   | 0.6092           | 0.8347  |
| Logistic Regression          | 0.7271   | 0.6082           | 0.8357  |

**Conclusion**: The **Random Forest** model emerged as the top performer after fine-tuning its hyperparameters, achieving the highest F1-score for identifying churners while maintaining a strong ROC-AUC.

---

## Key Visualizations

- **Confusion Matrix**: Shows the ability of Random Forest to correctly classify both churners and non-churners.  
- **ROC Curve**: Visualizes the trade-off between true positive rate and false positive rate across models.  
- **Feature Importance**: Highlights which factors most influenced predictions. Top features were consistently **Contract Type, Monthly Charges, and Tenure**.  
