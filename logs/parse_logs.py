import re
import pandas as pd

# Input log file
log_file = "pretrain_MLP/log_MLP32_16.txt"

# Output files
csv_output = "MLP.csv"
excel_output = "MLP.xlsx"

# Regex pattern:
# Handles BOTH:
#   Session 1 Iteration 0 ...
#   Iteration 0 ...
pattern = re.compile(
    r"(?:Session\s+(\d+)\s+)?Iteration\s+(\d+).*?"
    r"HR\s*=\s*([0-9.]+),\s*"
    r"NDCG\s*=\s*([0-9.]+),\s*"
    r"loss\s*=\s*([0-9.]+)"
)

data = []

# Default session number for single-run logs
default_session = 1

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            # If session exists, use it
            # Otherwise use default_session = 1
            session = (
                int(match.group(1))
                if match.group(1) is not None
                else default_session
            )

            iteration = int(match.group(2))
            hr = float(match.group(3))
            ndcg = float(match.group(4))
            loss = float(match.group(5))

            data.append({
                "Session": session,
                "Iteration": iteration,
                "HR@10": hr,
                "NDCG@10": ndcg,
                "Loss": loss
            })

# Create DataFrame
df = pd.DataFrame(data)

# Save files
df.to_csv(csv_output, index=False)
df.to_excel(excel_output, index=False)

print(df.head())

print(f"\nSaved CSV: {csv_output}")
print(f"Saved Excel: {excel_output}")