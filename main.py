import streamlit as st
import pandas as pd

st.title("AWS Lambda Cost Analysis Dashboard")

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader("Upload Lambda CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Raw Data")
st.dataframe(df)

# =======================================================
# Exercise 1: Identify top cost contributors (80% rule)
# =======================================================

st.header("Exercise 1: Top Cost Contributors")

df_sorted = df.sort_values("CostUSD", ascending=False)
df_sorted["CumulativeCost"] = df_sorted["CostUSD"].cumsum()
total_spend = df["CostUSD"].sum()
df_sorted["CumulativePct"] = df_sorted["CumulativeCost"] / total_spend

top_80 = df_sorted[df_sorted["CumulativePct"] <= 0.8]

st.subheader("Functions contributing to 80% of total spend")
st.dataframe(top_80[["FunctionName", "CostUSD", "CumulativePct"]])

st.info("Plot removed as requested.")

# =======================================================
# Exercise 2: Memory right-sizing
# =======================================================

st.header("Exercise 2: Memory Right-Sizing")

LOW_DURATION_MS = 100
HIGH_MEMORY_MB = 1024

memory_oversized = df[
    (df["AvgDurationMs"] < LOW_DURATION_MS) &
    (df["MemoryMB"] >= HIGH_MEMORY_MB)
]

st.subheader("Potential memory overprovision (low duration, high memory)")
st.dataframe(
    memory_oversized[["FunctionName", "AvgDurationMs", "MemoryMB", "CostUSD"]]
)

# ---- Cost Projection Function ----


def projected_cost(row, new_memory_mb):
    duration_sec = row["AvgDurationMs"] / 1000
    inv = row["InvocationsPerMonth"]

    old_gbsec = row["GBSeconds"]
    cost_per_gbsec = row["CostUSD"] / old_gbsec

    new_gbsec = (new_memory_mb / 1024) * duration_sec * inv
    return new_gbsec * cost_per_gbsec


if not memory_oversized.empty:
    memory_oversized["ProjectedCost_512MB"] = memory_oversized.apply(
        lambda r: projected_cost(r, 512), axis=1
    )

    st.subheader("Projected Cost if Memory Reduced to 512 MB")
    st.dataframe(
        memory_oversized[["FunctionName", "CostUSD", "ProjectedCost_512MB"]]
    )


# =======================================================
# Exercise 3: Provisioned Concurrency Optimization
# =======================================================

st.header("Exercise 3: Provisioned Concurrency Optimization")

pc = df[df["ProvisionedConcurrency"] > 0]

st.subheader("Functions with Provisioned Concurrency")
st.dataframe(
    pc[["FunctionName", "ProvisionedConcurrency", "ColdStartRate", "CostUSD"]]
)

st.write("**Guidance:** If ColdStartRate < 0.01, Provisioned Concurrency may not be needed.")


# =======================================================
# Exercise 4: Detect unused or low-value workloads
# =======================================================

st.header("Exercise 4: Detect Unused / Low-Value Workloads")

total_inv = df["InvocationsPerMonth"].sum()

low_value = df[
    (df["InvocationsPerMonth"] / total_inv < 0.01) &
    (df["CostUSD"] > df["CostUSD"].median())
]

st.subheader("Low invocation (<1%) but high cost workloads")
st.dataframe(low_value[["FunctionName", "InvocationsPerMonth", "CostUSD"]])


# =======================================================
# Exercise 5: Cost Forecasting Model
# =======================================================

st.header("Exercise 5: Cost Forecasting Model")

pricing_coeff = 0.0000000021  # simplified placeholder coefficient

df["ForecastCost"] = (
    df["InvocationsPerMonth"] *
    (df["AvgDurationMs"] / 1000) *
    (df["MemoryMB"] / 1024) *
    pricing_coeff
    + df["DataTransferGB"] * 0.08
)

st.subheader("Forecasted Cost Estimate")
st.dataframe(df[["FunctionName", "CostUSD", "ForecastCost"]])


# =======================================================
# Exercise 6: Containerization Candidates
# =======================================================

st.header("Exercise 6: Containerization Candidates")

container_candidates = df[
    (df["AvgDurationMs"] > 3000) &
    (df["MemoryMB"] > 2048) &
    (df["InvocationsPerMonth"] < 100000)
]

st.subheader("Long-running, high-memory, low-frequency functions")
st.dataframe(container_candidates[
    ["FunctionName", "AvgDurationMs", "MemoryMB", "InvocationsPerMonth", "CostUSD"]
])

st.success("Analysis Complete.")
