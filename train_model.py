import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
data = pd.read_csv("complaints.csv")

X = data["text"]
y_priority = data["priority"]
y_category = data["category"]

print("Splitting data into train and test sets...")
X_train, X_test, yp_train, yp_test, yc_train, yc_test = train_test_split(
    X, y_priority, y_category, test_size=0.2, random_state=42
)

print("Vectorizing text...")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Priority Model...")
model_priority = LogisticRegression(max_iter=1000)
model_priority.fit(X_train_vec, yp_train)

print("Training Category Model...")
model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train_vec, yc_train)

print("\n--- Model Evaluation ---")
# Evaluate Priority Model
yp_pred = model_priority.predict(X_test_vec)
print(f"Priority Model Accuracy: {accuracy_score(yp_test, yp_pred) * 100:.2f}%")
print("Priority Classification Report:")
print(classification_report(yp_test, yp_pred))

# Evaluate Category Model
yc_pred = model_category.predict(X_test_vec)
print(f"Category Model Accuracy: {accuracy_score(yc_test, yc_pred) * 100:.2f}%")
print("Category Classification Report:")
print(classification_report(yc_test, yc_pred))

print("\nSaving models to disk...")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model_priority.pkl", "wb") as f:
    pickle.dump(model_priority, f)

with open("model_category.pkl", "wb") as f:
    pickle.dump(model_category, f)

print("Models saved successfully! (vectorizer.pkl, model_priority.pkl, model_category.pkl)")
