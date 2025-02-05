import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# Baca file CSV
file_path = 'numerical_labels_data.csv'  # Ganti dengan path ke file CSV Anda
data = pd.read_csv(file_path)

# Pastikan kolom ada di CSV
if 'preprocessed_text' not in data.columns or 'label_sentiment' not in data.columns:
    raise ValueError("Kolom 'preprocessed_text' atau 'label_sentiment' tidak ditemukan dalam CSV!")

# Ambil teks dan label
texts = data['preprocessed_text'].astype(str)  # Pastikan kolom ini berupa string
labels = data['label_sentiment'].astype(int)  # Pastikan label berupa angka

# Preprocessing
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)  # 'texts' adalah data teks
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42, stratify=labels)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
rf_model.fit(X_train, y_train)

# Predicting
y_train_pred = rf_model.predict(X_train)
y_pred = rf_model.predict(X_test)

# Evaluating
training_accuracy = accuracy_score(y_train, y_train_pred)
testing_accuracy = accuracy_score(y_test, y_pred)
print(f"Training Accuracy: {round(training_accuracy, 5)}")
print(f"Testing Accuracy: {round(testing_accuracy, 5)}")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Not Complaint", "Complaint"], zero_division=0))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
validation_accuracy = round(np.mean(cv_scores), 5)
training_loss = round(1 - training_accuracy, 5)
validation_loss = round(1 - validation_accuracy, 5)

print(f"Validation Accuracy: {validation_accuracy}")
print(f"Training Loss: {training_loss}")
print(f"Validation Loss: {validation_loss}")

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Complaint", "Complaint"], yticklabels=["Not Complaint", "Complaint"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_test, y_pred, "Random Forest")

# Learning Curve and Proxy Loss
def plot_learning_curve_with_loss(model, X_train, y_train, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Compute loss as 1 - accuracy
    train_loss = 1 - train_scores_mean
    test_loss = 1 - test_scores_mean

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_scores_mean, label="Training Accuracy", color="blue")
    plt.plot(train_sizes, test_scores_mean, label="Validation Accuracy", color="green")
    plt.title(f"Training and Validation Accuracy - {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, train_loss, label="Training Loss", color="blue")
    plt.plot(train_sizes, test_loss, label="Validation Loss", color="green")
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.xlabel("Training Size")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

plot_learning_curve_with_loss(rf_model, X_train, y_train, "Random Forest")
