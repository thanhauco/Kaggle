import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine train and test data for preprocessing
all_data = pd.concat([train_data, test_data], sort=False)

# Feature engineering
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
all_data['Title'] = all_data['Title'].map(title_mapping)

# Fill missing age with median age for each title
all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# Create FamilySize feature
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# Create IsAlone feature
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

# Fill missing embarked with most frequent value
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])

# Fill missing Fare with median
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

# Create age bands
all_data['AgeBand'] = pd.cut(all_data['Age'], 5)

# Encode categorical variables
all_data = pd.get_dummies(all_data, columns=['Sex', 'Embarked', 'AgeBand'])

# Select features for model
features = ['Pclass', 'Title', 'FamilySize', 'IsAlone', 'Fare'] + \
           [col for col in all_data.columns if col.startswith(('Sex_', 'Embarked_', 'AgeBand_'))]

# Split back into train and test
train = all_data[:len(train_data)]
test = all_data[len(train_data):]

X = train[features]
y = train['Survived']

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions on validation set
val_predictions = rf_model.predict(X_val_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare test data
X_test = test[features]
X_test_scaled = scaler.transform(X_test)

# Make predictions on test set
test_predictions = rf_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("Submission file created.")
