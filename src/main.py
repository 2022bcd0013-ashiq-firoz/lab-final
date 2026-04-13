import pandas as pd

# Load the dataset
df = pd.read_csv("data/data.csv")

# Keep only the first 5000 rows
df_subset = df.head(5000)

# Save the updated version (optional: index=False prevents an extra column)
df_subset.to_csv("data/data_subset.csv", index=False)

print(f"Original size: {len(df)}")
print(f"New size: {len(df_subset)}")