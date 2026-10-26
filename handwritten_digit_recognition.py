
"""
Handwritten Digit Recognition using Python and Machine Learning
Dataset: scikit-learn built-in digits dataset (no download needed)

Install once:  pip install scikit-learn matplotlib joblib
Run:           python handwritten_digit_recognition.py
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # saves plots as images; remove this line to show plot windows
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 1. Load the dataset (1797 images of digits 0-9, each 8x8 pixels)
digits = load_digits()
X, y = digits.data, digits.target
print("Total images:", len(X))

# 2. Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the model (Support Vector Machine)
model = SVC(kernel="rbf", gamma=0.001, C=10)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 5. Save the confusion matrix as an image
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Show some sample predictions and save them as an image
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for ax, image, pred, actual in zip(axes.ravel(), X_test, y_pred, y_test):
    ax.imshow(image.reshape(8, 8), cmap="gray_r")
    ax.set_title(f"Pred: {pred} | True: {actual}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

# 7. Save the trained model so it can be reused later
joblib.dump(model, "digit_model.joblib")
print("Saved: digit_model.joblib, confusion_matrix.png, sample_predictions.png")
