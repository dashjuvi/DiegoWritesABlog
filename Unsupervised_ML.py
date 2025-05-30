import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (OPTICS, KMeans, AgglomerativeClustering, DBSCAN,
                            MeanShift, SpectralClustering, Birch,
                            estimate_bandwidth)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipaddress

# Load the JSON log file
file_path = r"yourpath"
with open(file_path, 'r', encoding='utf-8') as f:
    logs = json.load(f)

print(f"Original JSON records: {len(logs)}")

# Convert JSON to DataFrame
df = pd.DataFrame(logs)
print(f"Initial DataFrame rows: {len(df)}")

# Select relevant fields
fields_to_use = [
    'signature', 'source.ip', 'source.port',
    'windows.logonId', 'windows.taskCategory'
]

# Check missing values
for field in fields_to_use:
    missing_count = df[field].isna().sum()
    print(f"Missing values in {field}: {missing_count} ({missing_count/len(df)*100:.2f}%)")

# Print sample values
print("\nSample values before conversion:")
for field in fields_to_use:
    if field in df.columns:
        print(f"{field}: {df[field].head().tolist()}")

# Filter for rows with required fields
df_filtered = df.dropna(subset=fields_to_use)
print(f"\nAfter dropping NA in required fields: {len(df_filtered)}")

# Fallback to essential fields if needed
if len(df_filtered) == 0:
    print("No records with all required fields. Trying with fewer fields...")
    essential_fields = ['signature', 'source.ip', 'source.port']
    df_filtered = df.dropna(subset=essential_fields)
    print(f"After dropping NA in essential fields only: {len(df_filtered)}")

# Handle windows fields
if 'windows.logonId' in df.columns:
    df_filtered['windows.logonId'] = df_filtered['windows.logonId'].fillna('0x0')
if 'windows.taskCategory' in df.columns:
    df_filtered['windows.taskCategory'] = df_filtered['windows.taskCategory'].fillna('Unknown')

# IP conversion
def ip_to_int(ip):
    try:
        if pd.isna(ip):
            return np.nan
        return int(ipaddress.ip_address(str(ip)))
    except Exception as e:
        print(f"Failed to convert IP: {ip}, Error: {e}")
        return np.nan

df_filtered['source.ip'] = df_filtered['source.ip'].apply(ip_to_int)
print(f"After IP conversion: {len(df_filtered.dropna(subset=['source.ip']))}")

# LogonID conversion
def parse_logon_id(val):
    try:
        if pd.isna(val):
            return np.nan
        if isinstance(val, str) and val.startswith('0x'):
            return int(val, 16)
        return int(val)
    except Exception as e:
        print(f"Failed to convert logonId: {val}, Error: {e}")
        return np.nan

if 'windows.logonId' in df_filtered.columns:
    df_filtered['windows.logonId'] = df_filtered['windows.logonId'].apply(parse_logon_id)
    print(f"After logonId conversion: {len(df_filtered.dropna(subset=['windows.logonId']))}")

# One-hot encoding
if 'windows.taskCategory' in df_filtered.columns:
    print(f"Before one-hot encoding: {len(df_filtered)}")
    df_filtered = pd.get_dummies(df_filtered, columns=['windows.taskCategory'], drop_first=False)
    print(f"After one-hot encoding: {len(df_filtered)}")

# Final cleaning
print(f"Before dropping remaining NAs: {len(df_filtered)}")
df_filtered = df_filtered.dropna(subset=['signature', 'source.ip', 'source.port'])
print(f"After dropping remaining NAs: {len(df_filtered)}")

# Feature selection
features = ['signature', 'source.ip', 'source.port']
if 'windows.logonId' in df_filtered.columns:
    features.append('windows.logonId')
features += [col for col in df_filtered.columns if col.startswith('windows.taskCategory_')]

print(f"Selected features: {features}")
print(f"Number of features: {len(features)}")

# Check data availability
if len(df_filtered) == 0:
    print("ERROR: No data left after preprocessing!")
    exit()

# Prepare feature matrix
features = [f for f in features if f in df_filtered.columns]
X = df_filtered[features]
print(f"Shape of feature matrix: {X.shape}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate PCA projections
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Define clustering algorithms
clustering_algorithms = [
    ('K-Means', KMeans(n_clusters=3, random_state=42)),
    ('Hierarchical Ward', AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5)),
    ('Mean Shift', MeanShift(bandwidth=estimate_bandwidth(X_scaled, quantile=0.2))),
    ('Gaussian Mixture', GaussianMixture(n_components=3, random_state=42)),
    ('Spectral Clustering', SpectralClustering(n_clusters=3, random_state=42)),
    ('BIRCH', Birch(n_clusters=3)),
    ('Agglomerative Complete', AgglomerativeClustering(n_clusters=3, linkage='complete')),
    ('OPTICS', OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05))
]

# Compare all algorithms
for algo_name, algorithm in clustering_algorithms:
    try:
        print(f"\n=== Running {algo_name} ===")
        
        # Special handling for GMM
        if algo_name == 'Gaussian Mixture':
            gm = algorithm.fit(X_scaled)
            clusters = gm.predict(X_scaled)
        else:
            clusters = algorithm.fit_predict(X_scaled)
        
        # 2D Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.title(f'{algo_name} Clustering (2D PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(label='Cluster')
        plt.tight_layout()
        plt.show()

        # 3D Visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                            c=clusters, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        ax.set_title(f'{algo_name} Clustering (3D PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.tight_layout()
        plt.show()

        # Cluster statistics
        unique_clusters = np.unique(clusters)
        n_noise = np.sum(clusters == -1) if -1 in clusters else 0
        
        print(f"Clusters found: {len(unique_clusters) - (1 if -1 in clusters else 0)}")
        print(f"Noise points: {n_noise}")
        print("Cluster distribution:")
        for cluster in unique_clusters:
            count = np.sum(clusters == cluster)
            perc = count/len(clusters)*100
            if cluster == -1:
                print(f"  Noise: {count} ({perc:.1f}%)")
            else:
                print(f"  Cluster {cluster}: {count} ({perc:.1f}%)")
                
    except Exception as e:
        print(f"Error with {algo_name}: {str(e)}")
        continue

# Save results
output_path = r"output_path"
df_filtered.to_csv(output_path, index=False)
print(f"\nFinal results saved to {output_path}")
