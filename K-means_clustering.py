import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("K-means_clustering\Mall_Customers.csv")

# Drop unnecessary columns to focus on relevant features for clustering
df.drop(['CustomerID', 'Gender', 'Age'], axis=1, inplace=True)

# Prepare data for clustering
x = df.values

# Determine optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='purple')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(color='gray', linestyle=':', linewidth=0.5)
# plt.show()

# Apply K-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y = kmeans.fit_predict(x)

# Plotting clusters with enhanced colors and visuals
plt.figure(figsize=(12, 7))
sns.set_palette("Set2")

# Define colors for each cluster
colors = sns.color_palette("Set2", 5)
plt.scatter(x[y == 0, 0], x[y == 0, 1], s=60, color=colors[0], edgecolor='black', label='Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s=60, color=colors[1], edgecolor='black', label='Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s=60, color=colors[2], edgecolor='black', label='Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s=60, color=colors[3], edgecolor='black', label='Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s=60, color=colors[4], edgecolor='black', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='gold', marker='*', edgecolor='black', label='Centroids')

# Enhance plot appearance
plt.title("Customer Segments based on Annual Income and Spending Score", fontsize=14, fontweight='bold')
plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.legend(title="Clusters", fontsize=10, title_fontsize='13')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
sns.despine()

# Show plot
plt.show()
