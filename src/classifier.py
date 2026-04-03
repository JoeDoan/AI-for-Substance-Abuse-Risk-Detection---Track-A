import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib

# Load the stratified subset
df = pd.read_csv("data/stratified_subset.csv")

# Create a binary target
# 1 = High Risk (Severe distress/Substance Abuse)
# 0 = Safe / Lower Risk
high_risk_classes = ['addiction', 'alcoholism', 'suicidewatch', 'depression', 'ptsd', 'mentalhealth', 'anxiety']
df['is_high_risk'] = df['label'].apply(lambda x: 1 if x in high_risk_classes else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_high_risk'], test_size=0.2, random_state=42)

# Create a classification pipeline
# We use Logistic Regression for fast, robust baseline modeling
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

print("Training the high-precision Risk Classification model...")
pipeline.fit(X_train, y_train)

# Predict probabilities
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Prioritizing Recall over Precision (minimize false negatives)
# We lower the decision threshold to 0.4
THRESHOLD = 0.4
y_pred_precision = (y_proba >= THRESHOLD).astype(int)

print(f"\n--- Evaluation Metrics (Threshold = {THRESHOLD}) ---")
prec = precision_score(y_test, y_pred_precision)
rec = recall_score(y_test, y_pred_precision)
f1 = f1_score(y_test, y_pred_precision)

print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

report = classification_report(y_test, y_pred_precision, target_names=["Safe", "High Risk"])
print("\nClassification Report:\n", report)

# Save the model
joblib.dump(pipeline, "data/risk_classifier.pkl")
print("Saved classification model to data/risk_classifier.pkl")

# Save metrics for the Hackathon report
with open("data/evaluation_metrics.txt", "w") as f:
    f.write(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n\n")
    f.write(report)
