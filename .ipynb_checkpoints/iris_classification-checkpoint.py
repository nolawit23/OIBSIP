# iris_download_and_train.py
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Download the Iris dataset from Kaggle
print("⏳ Downloading Iris dataset from Kaggle...")
path = kagglehub.dataset_download("arshid/iris-flower-dataset")
print("✅ Dataset downloaded successfully!")

print("📁 Dataset folder path:", path)

# Step 2: Load the CSV file
csv_path = path + "/IRIS.csv"   # File name in the Kaggle dataset
df = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully!")

# Step 3: Display first few rows
print("\nSample data:")
print(df.head())

# Step 4: Prepare data for training
X = df.drop('species', axis=1)
y = LabelEncoder().fit_transform(df['species'])

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
print("\n🎯 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


sns.pairplot(df, hue='species')
plt.show()