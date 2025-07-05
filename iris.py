# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv("Iris.csv")

# Drop the 'Id' column as it's not useful for prediction
df.drop('Id', axis=1, inplace=True)

# Encode the target labels ('Species') into numeric values
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# Separate features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' treats all classes equally
recall = recall_score(y_test, y_pred, average='macro')

# Output results
print("Model Evaluation:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
