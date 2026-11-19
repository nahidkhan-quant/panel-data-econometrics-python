
import pyreadr

# Load the R dataset
result = pyreadr.read_r("SexWorkers.R")

# Display the available dataset names
print("Dataset names:", result.keys())

# Extract the first data frame
df = list(result.values())[0]

# Inspect the dataset
print("\nFirst five observations:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nVariable names:")
print(df.columns.tolist())

# Export as CSV
df.to_csv("SexWorkers.csv", index=False)

print("\nSexWorkers.csv created successfully!")
