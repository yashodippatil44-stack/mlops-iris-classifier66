import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv("data/raw/iris_v1.csv")

# Create random number generator
rng = np.random.default_rng(42)

# Generate small noise for 20 synthetic rows
noise = rng.normal(0, 0.05, size=(20, 4))

# Select 20 existing rows
sample = df.sample(20, random_state=42).reset_index(drop=True)

# Add noise to the first 4 columns
sample.iloc[:, :4] = sample.iloc[:, :4].values + noise

# Combine original + synthetic rows
augmented = pd.concat([df, sample], ignore_index=True)

# Save updated dataset
augmented.to_csv("data/raw/iris_v1.csv", index=False)

print(f"Dataset now has {len(augmented)} rows")