import io
import csv
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Cloud Cost Tag Governance Simulator", layout="wide")

# =========================================================
# Forgiving CSV Loader (handles quoted-whole-line CSV too)
# =========================================================
def load_csv_forgiving(uploaded_file_or_path):
    def _read(obj):
        return pd.read_csv(obj, low_memory=False)

    try:
        df = _read(uploaded_file_or_path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    if hasattr(uploaded_file_or_path, "getvalue"):
        raw = uploaded_file_or_path.getvalue().decode("utf-8", errors="replace")
    else:
        with open(uploaded_file_or_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

    fixed_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        fixed_lines.append(s)
    fixed_text = "\n".join(fixed_lines)
    buf = io.StringIO(fixed_text)
    return pd.read_csv(buf, low_memory=False)

# =========================================================
# Schema helpers
# =========================================================
REQUIRED_COLUMNS = [
    "ResourceID","Service","Region","Department","Project","Environment","Owner",
    "CostCenter","CreatedBy","MonthlyCostUSD","Tagged"
]
TAG_FIELDS = ["Department","Project","Environment","Owner","CostCenter","CreatedBy"]

def ensure_cost_allocation_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["Tagged"] = df["Tagged"].astype(str).str.strip().str.title()  # 'Yes'/'No'
    df["Service"] = df["Service"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.strip()
    df["MonthlyCostUSD"] = pd.to_numeric(df["MonthlyCostUSD"], errors="coerce")
    return df

def tag_completeness_score(row: pd.Series) -> int:
    score = 0
    for f in TAG_FIELDS:
        v = row.get(f, None)
        if pd.notna(v) and str(v).strip() != "":
            score += 1
    return score

def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TagCompletenessScore"] = df.apply(tag_completeness_score, axis=1)
    df["IsUntagged"] = df["Tagged"].astype(str).str.title().eq("No")
    return df

def fmt_pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "—"

# =========================================================
# Load dataset
# =========================================================
st.title("💸 Cloud Cost Tag Governance Simulator (INFO49971)")

uploaded_file = st.file_uploader("Upload CloudMart Resource Tagging Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_raw = load_csv_forgiving(uploaded_file)
else:
    st.info("No file uploaded. You can still explore the app with a tiny example.")
    demo_csv = io.StringIO(
        """AccountID,ResourceID,Service,Region,Department,Project,Environment,Owner,CostCenter,CreatedBy,MonthlyCostUSD,Tagged
1001,i-001,EC2,us-east-1,Marketing,CampaignApp,Prod,j.smith@cloudmart.com,CC101,Terraform,120,Yes
1001,i-002,EC2,us-east-1,Marketing,CampaignApp,Dev,,CC101,Terraform,80,No
1001,s3-001,S3,us-east-1,Marketing,AdsAPI,Prod,j.smith@cloudmart.com,CC101,Jenkins,60,Yes
1001,s3-002,S3,us-east-1,Marketing,AdsAPI,Dev,,CC101,Manual,25,No
"""
    )
    df_raw = pd.read_csv(demo_csv)

df = ensure_cost_allocation_schema(df_raw)
df = add_computed_columns(df)

# --- FIXED: update session state when new file uploaded ---
def _fingerprint_from_upload(up):
    if up is None:
        return "demo"
    try:
        b = up.getvalue()
    except Exception:
        b = b""
    return "upload:" + hashlib.md5(b).hexdigest()

data_fp = _fingerprint_from_upload(uploaded_file)
if "data_fingerprint" not in st.session_state or st.session_state.get("data_fingerprint") != data_fp:
    st.session_state.data_fingerprint = data_fp
    st.session_state.df_original = df.copy()
    st.session_state.df_remediated = df.copy()
    st.toast("✅ Loaded dataset successfully!", icon="📂")

# =========================================================
# Helper functions
# =========================================================
def apply_filters(dfin: pd.DataFrame, f_service, f_region, f_dept, f_env) -> pd.DataFrame:
    d = dfin.copy()
    if f_service != "(All)":
        d = d[d["Service"] == f_service]
    if f_region != "(All)":
        d = d[d["Region"] == f_region]
    if f_dept != "(All)":
        d = d[d["Department"] == f_dept]
    if f_env != "(All)":
        d = d[d["Environment"] == f_env]
    return d

def tagging_metrics(dfin: pd.DataFrame):
    d = dfin.copy()
    total = len(d)
    tagged_counts = d["Tagged"].value_counts(dropna=False)
    n_tagged = int(tagged_counts.get("Yes", 0))
    n_untagged = int(tagged_counts.get("No", 0))
    pct_untagged = (n_untagged / total * 100.0) if total > 0 else np.nan
    total_cost = pd.to_numeric(d["MonthlyCostUSD"], errors="coerce").sum()
    cost_by_tag = d.groupby("Tagged", dropna=False)["MonthlyCostUSD"].sum(min_count=1)
    untagged_cost = float(cost_by_tag.get("No", 0.0))
    pct_untagged_cost = (untagged_cost / total_cost * 100.0) if total_cost > 0 else np.nan
    return {
        "total": total,
        "n_tagged": n_tagged,
        "n_untagged": n_untagged,
        "pct_untagged": pct_untagged,
        "total_cost": total_cost,
        "untagged_cost": untagged_cost,
        "pct_untagged_cost": pct_untagged_cost,
    }

# =========================================================
# Global filters
# =========================================================
with st.expander("🔎 Global Filters"):
    df_all = st.session_state.df_remediated
    svc_opts = ["(All)"] + sorted(df_all["Service"].dropna().unique().tolist())
    reg_opts = ["(All)"] + sorted(df_all["Region"].dropna().unique().tolist())
    dept_opts = ["(All)"] + sorted(df_all["Department"].dropna().unique().tolist())
    env_opts = ["(All)"] + sorted(df_all["Environment"].dropna().unique().tolist())
    f_service = st.selectbox("Service", svc_opts, index=0)
    f_region = st.selectbox("Region", reg_opts, index=0)
    f_dept = st.selectbox("Department", dept_opts, index=0)
    f_env = st.selectbox("Environment", env_opts, index=0)

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    "1) Data Exploration",
    "2) Cost Visibility",
    "3) Tagging Compliance",
])

# ------------------------------
# Tab 1: Data Exploration
# ------------------------------
with tabs[0]:
    st.header("Task Set 1 – Data Exploration")
    df_f = apply_filters(st.session_state.df_remediated, f_service, f_region, f_dept, f_env)
    st.subheader("First 5 Rows")
    st.dataframe(df_f.head(), use_container_width=True)
    st.subheader("Missing Values per Column")
    st.write(df_f.isnull().sum())
    m = tagging_metrics(df_f)
    st.metric("Untagged %", fmt_pct(m["pct_untagged"]))

# ------------------------------
# Tab 2: Cost Visibility
# ------------------------------
with tabs[1]:
    st.header("Task Set 2 – Cost Visibility")
    df_f = apply_filters(st.session_state.df_remediated, f_service, f_region, f_dept, f_env)
    m = tagging_metrics(df_f)
    st.metric("Total Cost", f"${m['total_cost']:,.2f}")
    st.metric("Untagged Cost", f"${m['untagged_cost']:,.2f}")
    st.metric("% Untagged Cost", fmt_pct(m["pct_untagged_cost"]))
    cost_by_tag = df_f.groupby("Tagged", dropna=False)["MonthlyCostUSD"].sum(min_count=1).reset_index()
    st.dataframe(cost_by_tag)
    fig = px.pie(cost_by_tag, names="Tagged", values="MonthlyCostUSD", title="Cost Share by Tagging")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 3: Tagging Compliance
# ------------------------------
with tabs[2]:
    st.header("Task Set 3 – Tagging Compliance")
    df_f = apply_filters(st.session_state.df_remediated, f_service, f_region, f_dept, f_env)
    df_f["TagCompletenessScore"] = df_f.apply(tag_completeness_score, axis=1)
    st.dataframe(
        df_f[["ResourceID","Service","Department","Project","Environment","Owner","CostCenter","CreatedBy","TagCompletenessScore"]],
        use_container_width=True
    )
    st.download_button(
        "Download Full Processed Dataset (All Rows)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="cloudmart_processed.csv",
        mime="text/csv"
    )

st.caption("✅ Fixed version: loads *all rows* from uploaded CSV and updates session state.")
