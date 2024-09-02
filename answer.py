import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, entropy

# Load the data
df = pd.read_csv('cancer.csv')

# Display initial data information
print(df.info())
print(df.describe())

# Convert the 'Diagnosis' column from textual to numerical values
df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

# Calculate and display entropy for each feature
features = df.columns[:-1]  # The last column is the diagnosis status
entropies = {feature: entropy(df[feature].value_counts(normalize=True)) for feature in features}

print("Entropies of features:")
for feature, value in entropies.items():
    print(f"{feature}: {value:.4f}")

# Calculate mutual information
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

mutual_info = mutual_info_classif(X, y)
mutual_info_series = pd.Series(mutual_info, index=X.columns)
mutual_info_series.sort_values(ascending=False).plot.bar(figsize=(12, 6), color='teal')
plt.title("Mutual Information of Features")
plt.show()

# Plot distributions for each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, bins=30, color='purple')
    plt.title(f'Distribution of {feature}')
    plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to the data
df['Cluster'] = clusters

# Display clustering results
sns.pairplot(df, hue='Cluster', vars=features[:4])
plt.show()

# Calculate the correlation matrix
correlation_matrix = df[features].corr(method='spearman')

# Display the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Spearman Correlation Matrix')
plt.show()

# Create a correlation map based on the mean values of each feature
mean_values = df.groupby('Diagnosis').mean()
mean_corr_matrix = mean_values.corr(method='spearman')

plt.figure(figsize=(12, 8))
sns.heatmap(mean_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Map of Mean Values')
plt.show()
