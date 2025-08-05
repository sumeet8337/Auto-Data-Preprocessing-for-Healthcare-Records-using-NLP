import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset.csv')

# Gender Encoding: Male=10, Female=11, Other=12
df['Gender'] = df['Gender'].map({'Male': 10, 'Female': 11}).fillna(12).astype(int)

# Age Cleaning
df['Age'] = df['Age'].apply(lambda x: np.nan if x < 0 else x)

# Age Category: Child=30, Adult=31, Senior=32, Unknown=33
def categorize_age(age):
    if pd.isna(age): return 33
    elif age <= 12: return 30
    elif 13 <= age <= 59: return 31
    elif age >= 60: return 32

df['Age_Category'] = df['Age'].apply(categorize_age)

# Diagnosis Mapping: Flu=20, COVID=21, Cancer=22, Cold=23
diagnosis_mapping = {'Flu': 20, 'COVID-19': 21, 'Cancer': 22, 'Cold': 23}
df['Diagnosis'] = df['Diagnosis'].map(diagnosis_mapping).fillna(-1).astype(int)

# Medicine Assignment: Low=40, Medium=41, High=42
def assign_medicine(age_category):
    if age_category == 30: return 40  # Low dose
    elif age_category == 31: return 42  # High dose
    elif age_category == 32: return 40  # Low dose
    elif age_category == 33: return 41  # Medium dose

df['Medicine'] = df['Age_Category'].apply(assign_medicine)

# 1. Medicine Recommendation

X = df[['Age', 'Gender', 'Diagnosis']]
y = df['Medicine']

model = DecisionTreeClassifier()
model.fit(X, y)

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=['Low', 'Medium', 'High'], filled=True)
plt.title("Decision Tree for Medicine Recommendation")
plt.savefig("decision_tree.png")
plt.close()

# 2. Disease Risk Prediction

df['Blood_Pressure'] = np.random.randint(80, 180, size=len(df))
df['Sugar_Level'] = np.random.randint(70, 200, size=len(df))

X_risk = df[['Age', 'Gender', 'Diagnosis', 'Blood_Pressure', 'Sugar_Level']]
df['Risk_Level'] = np.where((df['Blood_Pressure'] > 140) | (df['Sugar_Level'] > 160), 'High',
                    np.where((df['Blood_Pressure'] > 120) | (df['Sugar_Level'] > 130), 'Medium', 'Low'))

risk_model = RandomForestClassifier()
risk_model.fit(X_risk, df['Risk_Level'])

# 3. Treatment Timeline Generator

def generate_timeline(row):
    diag = row['Diagnosis']
    if diag == 20:  # Flu
        return "Day 1-3: Med A, Day 4-7: Med B, Day 8: Review"
    elif diag == 21:  # COVID
        return "Day 1-5: Med C, Day 6-10: Med D"
    elif diag == 22:  # Cancer
        return "Day 1-7: Med E, Day 8-14: Med F, Day 15: chemotherapy"
    elif diag == 23:  # Cold
        return "Day 1-2: Med G, Day 3: Med H"
    else:
        return "Consult doctor"

df['Treatment_Timeline'] = df.apply(generate_timeline, axis=1)

df.to_csv('final_healthcare_dataset.csv', index=False, na_rep='NaN')
