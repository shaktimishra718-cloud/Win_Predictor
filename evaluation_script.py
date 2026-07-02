#  NEW: PERFORMANCE EVALUATION 
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X)

print("\n--- Model Performance Report ---")
print(f"Overall Accuracy: {accuracy_score(y, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y, y_pred))

importances = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_})
print("\n--- Feature Importance ---")
print(importances.sort_values(by='importance', ascending=False))


print("Model trained on Form Differential and Relative Ranking!")
