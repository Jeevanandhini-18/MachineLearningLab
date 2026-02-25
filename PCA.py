# ================================
# Principal Component Analysis
# Online Shoppers Dataset
# ================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")
print("Original Dataset:\n", df.head())

# Step 3: Select Only Numerical Columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Remove target column 'Revenue' if present
if 'Revenue' in numeric_df.columns:
    numeric_df = numeric_df.drop('Revenue', axis=1)

X = numeric_df
print("\nShape of Feature Matrix:", X.shape)

# =====================================
# -------- MANUAL PCA -----------------
# =====================================

# Step 4: Standardize Data (Manual)
mean = X.mean()
std = X.std()
Z = (X - mean) / std
print("\nStandardized Data (Manual):\n", Z.head())

# Step 5: Covariance Matrix
cov_matrix = np.cov(Z.T)
print("\nCovariance Matrix:\n", cov_matrix)

# Step 6: Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)

# Step 7: Sort Eigenvalues in Descending Order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("\nSorted Eigenvalues:\n", eigenvalues)

# Step 8: Explained Variance Ratio (Manual)
explained_variance = eigenvalues / np.sum(eigenvalues)
print("\nExplained Variance Ratio (Manual):\n", explained_variance)

# Step 9: Project onto First 2 Principal Components
PCs = eigenvectors[:, :2]
pca_manual = np.dot(Z, PCs)
print("\nPCA Result (Manual - First 5 Rows):\n", pca_manual[:5])

# =====================================
# -------- SKLEARN PCA ---------------
# =====================================

# Step 10: Standardize using sklearn
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(X)

# Step 11: Apply PCA (2 Components)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(Z_scaled)

print("\nExplained Variance Ratio (Sklearn):")
print(pca.explained_variance_ratio_)

print("\nPCA Result (Sklearn - First 5 Rows):\n", pca_result[:5])

# =====================================
# -------- Visualization --------------
# =====================================

plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Online Shoppers Dataset")
plt.show()