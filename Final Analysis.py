import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = os.path.expanduser("~/Desktop/ad_figures")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/Users/thinurishehara/Desktop/combined_features_with_demographics.csv")
df.columns = df.columns.str.strip()

texture_features = [col for col in df.columns if col.startswith("GLCM_")]
volume_features = ['Volume_mm3', 'Volume_cm3', 'eTIV', 'nWBV', 'ASF']
demographic_features = ['Age', 'EDUC', 'SES', 'M/F', 'Hand']
all_features = texture_features + volume_features + demographic_features
key_features = all_features + ['Group']

df = df.dropna(subset=key_features)
df_binary = df[df['Group'].isin(['Demented', 'Nondemented'])].copy()

le = LabelEncoder()
df_binary['Group_encoded'] = le.fit_transform(df_binary['Group'])
for col in demographic_features:
    if df_binary[col].dtype == 'object':
        df_binary[col] = LabelEncoder().fit_transform(df_binary[col].astype(str))

zero_frac = (df_binary[texture_features + volume_features] == 0).sum(axis=1) / (len(texture_features) + len(volume_features))
df_binary = df_binary[zero_frac < 0.9]

feature_groups = {
    'GLCM': texture_features,
    'Volume': volume_features,
    'Demographics': demographic_features
}

def plot_confusion_matrix_percent(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums.astype(float) * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (%) - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

y = df_binary['Group_encoded']
combo_results = {}
roc_curves = {}

for r in range(1, len(feature_groups) + 1):
    for combo in combinations(feature_groups.keys(), r):
        combo_name = " + ".join(combo)
        selected_features = []
        for grp in combo:
            selected_features.extend(feature_groups[grp])
        X = df_binary[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, sampling_strategy=1.0)),
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        param_grid = {
            'rf__n_estimators': [100],
            'rf__max_depth': [None],
            'rf__min_samples_split': [2],
            'rf__bootstrap': [True]
        }

        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[combo_name] = (fpr, tpr, auc)

        cm_filename = os.path.join(output_dir, f"{combo_name.replace(' ', '_')}_confusion_matrix.png")
        plot_confusion_matrix_percent(y_test, y_pred, le.classes_, combo_name, cm_filename)

        combo_results[combo_name] = {'Accuracy': acc, 'AUC': auc}

summary_df = pd.DataFrame(combo_results).T.sort_values(by='Accuracy', ascending=False)
barplot_path = os.path.join(output_dir, "summary_accuracy_auc_barplot.png")
plt.figure(figsize=(12, 5))
summary_df[['Accuracy', 'AUC']].plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Comparison of Accuracy and AUC Across Feature Sets')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(barplot_path, dpi=300)
plt.close()

roc_path = os.path.join(output_dir, "roc_curves.png")
plt.figure(figsize=(8, 6))
for label, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Feature Combinations")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(roc_path, dpi=300)
plt.close()

