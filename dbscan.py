import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("station_month.csv")

# Select relevant columns and drop missing values
X = df[['Day', 'Night']].dropna()

# Scale data for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
clusters = db.fit_predict(X_scaled)

# Add cluster labels back to DataFrame
df_clean = X.copy()
df_clean['cluster'] = clusters

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(df_clean['Day'], df_clean['Night'], 
            c=df_clean['cluster'], cmap='viridis', s=50)
plt.xlabel("Day Noise Level (dB)")
plt.ylabel("Night Noise Level (dB)")
plt.title("DBSCAN Clustering of Noise Levels")
plt.colorbar(label='Cluster ID')
plt.show()

# Save clustered data
df_clean.to_csv("noise_clusters.csv", index=False)
print("Clustered data saved as noise_clusters.csv")
