import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

# Load the dataset
df_train = pd.read_csv('df_train.csv')  # Update the path if needed

# Columns to drop
columns_to_drop = [
    'Customer_ID', 
    'Month',                   
    'Annual_Income',           
    'Monthly_Balance',   
    'Changed_Credit_Limit',    
    'Credit_Utilization_Ratio', 
    'Total_EMI_per_month',     
    'Amount_invested_monthly', 
]

# Split data into features (X) and target (y)
X_train = df_train.drop(columns=columns_to_drop + ['Credit_Score'])
y_train = df_train['Credit_Score']

# Perform one-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train)

# Apply random oversampling to balance the classes in y_train
train_data = pd.concat([X_train_encoded, y_train], axis=1)
minority_class = train_data[train_data['Credit_Score'] == 'Poor']
majority_class = train_data[train_data['Credit_Score'] != 'Poor']
minority_class_upsampled = resample(minority_class,
                                    replace=True,     
                                    n_samples=len(majority_class),    
                                    random_state=42)
train_data_balanced = pd.concat([majority_class, minority_class_upsampled])

# Split the data back into X and y
X_train_resampled = train_data_balanced.drop(columns='Credit_Score')
y_train_resampled = train_data_balanced['Credit_Score']

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train_resampled, y_train_resampled)

# Save the trained model to a file
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model has been saved as 'random_forest_model.pkl'.")

# Save the columns used for training the model
X_train_resampled.columns.to_list()
joblib.dump(X_train_resampled.columns.to_list(), 'columns_used.pkl')