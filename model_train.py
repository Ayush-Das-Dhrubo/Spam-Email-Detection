import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("enron_dataset.csv", encoding="latin-1")

# We'll combine Subject + Message as text
df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')

# Convert labels to numeric
df['label_num'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})

# -------------------------------
# 2️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label_num'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 3️⃣ TF-IDF Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# -------------------------------
# 4️⃣ Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train_tf, y_train)

# -------------------------------
# 5️⃣ Evaluate
# -------------------------------
preds = model.predict(X_test_tf)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc*100:.2f}%")

# -------------------------------
# 6️⃣ Save Model & Vectorizer
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model & Vectorizer saved successfully!")
