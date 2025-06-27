import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

features_path = "/Users/thinurishehara/Desktop/oas2_masks2/brain_texture_summary.csv"
demographics_path = "/Users/thinurishehara/Desktop/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"

features_df = pd.read_csv(features_path)
demo_df = pd.read_excel(demographics_path)

print("\n=== Initial Data Shapes ===")
print("Features data shape:", features_df.shape)
print("Demographics data shape:", demo_df.shape)

features_df['MRI_ID'] = features_df['Subject_ID'].str.extract(r'(OAS2_\d{4}_MR\d)')[0]
demo_df['MRI_ID'] = demo_df['MRI ID'].str.strip()

print("\n=== Duplicate Check ===")
print("Duplicate MRI_IDs in features data:", features_df['MRI_ID'].duplicated().sum())
print("Duplicate MRI_IDs in demographics data:", demo_df['MRI_ID'].duplicated().sum())

dup_features = features_df[features_df['MRI_ID'].duplicated(keep=False)].sort_values('MRI_ID')
print("\nDuplicate feature entries:\n", dup_features[['MRI_ID', 'Filename']].head())

features_df = features_df.drop_duplicates(subset='MRI_ID', keep='first')

merged_df = pd.merge(
    features_df,
    demo_df,
    how='inner',
    on='MRI_ID'
)

print("\n=== Merged Data Shape ===")
print("Merged data shape:", merged_df.shape)

print("\n=== Missing Values in Merged Data ===")
print(merged_df.isnull().sum())

merged_df = merged_df.dropna(subset=['Mean_Intensity', 'MMSE'])

print("\n=== Basic Statistics ===")
print(merged_df.describe())

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Group', y='Volume_cm3', data=merged_df)
plt.title('Brain Volume by Group')

plt.subplot(2, 2, 2)
sns.histplot(merged_df['Age'], bins=20, kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 3)
sns.boxplot(x='M/F', y='GLCM_Contrast', data=merged_df)
plt.title('Texture Contrast by Gender')

plt.subplot(2, 2, 4)
corr_cols = ['Volume_cm3', 'GLCM_Contrast', 'GLCM_Energy', 'Age', 'MMSE']
sns.heatmap(merged_df[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')

plt.tight_layout()
plt.savefig("/Users/thinurishehara/Desktop/oas2_masks/initial_analysis.png")
plt.close()

print("\n=== Statistical Tests ===")

if 'Nondemented' in merged_df['Group'].values and 'Demented' in merged_df['Group'].values:
    nondemented = merged_df[merged_df['Group'] == 'Nondemented']['Volume_cm3']
    demented = merged_df[merged_df['Group'] == 'Demented']['Volume_cm3']
    t_stat, p_val = stats.ttest_ind(nondemented, demented)
    print(f"Volume comparison (Nondemented vs Demented): t={t_stat:.2f}, p={p_val:.4f}")

corr, p_val = stats.pearsonr(merged_df['Age'], merged_df['Volume_cm3'])
print(f"Age-Volume correlation: r={corr:.2f}, p={p_val:.4f}")

output_path = "/Users/thinurishehara/Desktop/oas2_masks2/merged_features_with_demographics2.csv"
merged_df.to_csv(output_path, index=False)

