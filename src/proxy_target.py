import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load your data
df = pd.read_csv(r"data/raw/data.csv")

######################### RFM Metrics  ######################################

# Ensure TransactionStartTime is datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Define snapshot date
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
print("Snapshot Date:", snapshot_date)

# Group by CustomerId
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
    'TransactionId': 'count',                                          # Frequency
    'Amount': 'sum'                                                    # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

######################### Cluster Customers ######################################

# Scale RFM Features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Sort clusters to understand behavior
cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
print("Cluster RFM Summary:\n", cluster_profile)

# Cluster 0: Recent, frequent, high spenders → Best Customers
# Cluster 1: Long ago, rare, low spenders → At Risk
# Cluster 2: Somewhat active, moderate value → Potential Loyalists

######################### Define and Assign the "High-Risk" Label ######################################

high_risk_cluster = cluster_profile['Recency'].idxmax()  # or manually set to 1 based on analysis
print("High-risk cluster identified as:", high_risk_cluster)

rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
print(rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'is_high_risk']].head())
rfm.to_csv("data/processed/rfm.csv", index=False)

######################### Integrate the Target Variable ######################################

merged_df = df.merge(
    rfm[['CustomerId', 'is_high_risk']],
    on='CustomerId',
    how='left'
)
merged_df.to_csv("data/processed/merged_df.csv", index=False)
