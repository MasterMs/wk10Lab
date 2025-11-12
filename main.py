import io
import csv
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
    """Create missing columns if absent so the rest of the app never breaks."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    # Normalize values
    df["Tagged"] = df["Tagged"].astype(str).str.strip().str.title()  # 'Yes'/'No'
    df["Service"] = df["Service"].astype(str).str.strip()
    df["Region"] = df["Region"].astype(str).str.strip()
    # Numeric cost
    df["MonthlyCostUSD"] = pd.to_numeric(df["MonthlyCostUSD"], errors="coerce")
    return df

def tag_completeness_score(row: pd.Series) -> int:
    """Count non-empty tag fields for the resource."""
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

if "df_remediated" not in st.session_state:
    st.session_state.df_original = df.copy()
    st.session_state.df_remediated = df.copy()

# =========================================================
# Global filters (helpful for all tabs)
# =========================================================
with st.expander("🔎 Global Filters"):
    svc_opts = ["(All)"] + sorted([x for x in df["Service"].dropna().unique()])
    reg_opts = ["(All)"] + sorted([x for x in df["Region"].dropna().unique()])
    dept_opts = ["(All)"] + sorted([x for x in df["Department"].dropna().unique()])
    env_opts = ["(All)"] + sorted([x for x in df["Environment"].dropna().unique()])

    f_service = st.selectbox("Service", svc_opts, index=0)
    f_region = st.selectbox("Region", reg_opts, index=0)
    f_dept = st.selectbox("Department", dept_opts, index=0)
    f_env = st.selectbox("Environment", env_opts, index=0)

def apply_filters(dfin: pd.DataFrame) -> pd.DataFrame:
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

# =========================================================
# Helper metrics
# =========================================================
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
# Tabs
# =========================================================
tabs = st.tabs([
    "1) Data Exploration",
    "2) Cost Visibility",
    "3) Tagging Compliance",
    "4) Visualization Dashboard",
    "5) Tag Remediation",
    "6) Downloads & Report",
])

# ------------------------------
# Tab 1: Data Exploration (Task Set 1)
# ------------------------------
with tabs[0]:
    st.header("Task Set 1 – Data Exploration")

    df_f = apply_filters(st.session_state.df_remediated)

    st.subheader("1.1 First 5 Rows")
    st.dataframe(df_f.head(), use_container_width=True)

    st.subheader("1.2 Missing Values by Column")
    st.write(df_f.isnull().sum())

    st.subheader("1.3 Columns with Most Missing Values")
    missing_sorted = df_f.isnull().sum().sort_values(ascending=False)
    st.write(missing_sorted)

    st.subheader("1.4 Tagged vs Untagged Counts")
    counts = df_f["Tagged"].value_counts(dropna=False)
    st.write(counts)

    st.subheader("1.5 % of Resources Untagged")
    m = tagging_metrics(df_f)
    st.metric("% Untagged Resources", fmt_pct(m["pct_untagged"]))

# ------------------------------
# Tab 2: Cost Visibility (Task Set 2)
# ------------------------------
with tabs[1]:
    st.header("Task Set 2 – Cost Visibility")

    df_f = apply_filters(st.session_state.df_remediated)
    m = tagging_metrics(df_f)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Cost", f"${m['total_cost']:,.2f}")
    with c2:
        st.metric("Untagged Cost", f"${m['untagged_cost']:,.2f}")
    with c3:
        st.metric("% Cost Untagged", fmt_pct(m["pct_untagged_cost"]))

    st.subheader("2.1 / 2.2 Tagged vs Untagged Cost")
    cost_by_tag = (
        df_f.groupby("Tagged", dropna=False)["MonthlyCostUSD"]
        .sum(min_count=1).reset_index()
    )
    st.dataframe(cost_by_tag, use_container_width=True)
    fig = px.pie(cost_by_tag, names="Tagged", values="MonthlyCostUSD", title="Cost Share by Tagging")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2.3 Department with Most Untagged Cost")
    dep_untag = (
        df_f[df_f["Tagged"].eq("No")]
        .groupby("Department", dropna=False)["MonthlyCostUSD"]
        .sum(min_count=1).reset_index()
        .sort_values("MonthlyCostUSD", ascending=False)
    )
    st.dataframe(dep_untag.head(10), use_container_width=True)

    st.subheader("2.4 Project with Highest Overall Cost")
    proj_cost = (
        df_f.groupby("Project", dropna=False)["MonthlyCostUSD"]
        .sum(min_count=1).reset_index()
        .sort_values("MonthlyCostUSD", ascending=False)
    )
    st.dataframe(proj_cost.head(10), use_container_width=True)

    st.subheader("2.5 Prod vs Dev (Cost & Tagging)")
    env_tag = (
        df_f.groupby(["Environment","Tagged"], dropna=False)["MonthlyCostUSD"]
        .sum(min_count=1).reset_index()
    )
    st.dataframe(env_tag, use_container_width=True)
    fig_env = px.bar(
        env_tag, x="Environment", y="MonthlyCostUSD", color="Tagged",
        barmode="group", title="Cost by Environment & Tagging"
    )
    st.plotly_chart(fig_env, use_container_width=True)

# ------------------------------
# Tab 3: Tagging Compliance (Task Set 3)
# ------------------------------
with tabs[2]:
    st.header("Task Set 3 – Tagging Compliance")

    df_f = apply_filters(st.session_state.df_remediated)

    st.subheader("3.1 Tag Completeness Score per Resource")
    st.caption("Score counts non-empty fields among: Department, Project, Environment, Owner, CostCenter, CreatedBy.")
    st.dataframe(
        df_f[["ResourceID","Service","Department","Project","Environment","Owner","CostCenter","CreatedBy","TagCompletenessScore"]],
        use_container_width=True
    )

    st.subheader("3.2 Top 5 Lowest Completeness")
    low5 = df_f.sort_values("TagCompletenessScore", ascending=True).head(5)
    st.dataframe(low5, use_container_width=True)

    st.subheader("3.3 Most Frequently Missing Tag Fields")
    missing_counts = {}
    for f in TAG_FIELDS:
        missing_counts[f] = int(df_f[f].isna().sum() + (df_f[f].astype(str).str.strip() == "").sum())
    miss_df = pd.DataFrame({"Field": list(missing_counts.keys()), "MissingCount": list(missing_counts.values())}).sort_values("MissingCount", ascending=False)
    st.dataframe(miss_df, use_container_width=True)

    st.subheader("3.4 Untagged Resources & Their Costs")
    untagged = df_f[df_f["Tagged"].eq("No")][["ResourceID","Service","Region","Department","Project","Environment","Owner","MonthlyCostUSD"]]
    st.dataframe(untagged, use_container_width=True)

    st.subheader("3.5 Export Untagged Resources")
    st.download_button(
        "Download untagged.csv",
        data=untagged.to_csv(index=False).encode("utf-8"),
        file_name="untagged.csv",
        mime="text/csv"
    )

# ------------------------------
# Tab 4: Visualization Dashboard (Task Set 4)
# ------------------------------
with tabs[3]:
    st.header("Task Set 4 – Visualization Dashboard")

    df_f = apply_filters(st.session_state.df_remediated)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("4.1 Tagged vs Untagged (Count)")
        pie_counts = (
            df_f.assign(One=1)
            .groupby("Tagged")["One"].sum()
            .reset_index()
            .rename(columns={"One":"Count"})
        )
        fig_pie = px.pie(pie_counts, names="Tagged", values="Count", title="Resources by Tagging")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("4.4 Cost by Environment")
        env_cost = df_f.groupby("Environment", dropna=False)["MonthlyCostUSD"].sum(min_count=1).reset_index()
        fig_env_cost = px.bar(env_cost, x="Environment", y="MonthlyCostUSD", title="Monthly Cost by Environment")
        st.plotly_chart(fig_env_cost, use_container_width=True)

    st.subheader("4.2 Cost per Department by Tagging")
    dept_tag_cost = (
        df_f.groupby(["Department","Tagged"], dropna=False)["MonthlyCostUSD"]
        .sum(min_count=1).reset_index()
    )
    fig_dept = px.bar(dept_tag_cost, x="Department", y="MonthlyCostUSD", color="Tagged", barmode="group",
                      title="Department Cost by Tagging")
    st.plotly_chart(fig_dept, use_container_width=True)

    st.subheader("4.3 Total Cost per Service (Horizontal)")
    svc_cost = df_f.groupby("Service", dropna=False)["MonthlyCostUSD"].sum(min_count=1).reset_index()
    fig_svc = px.bar(svc_cost, x="MonthlyCostUSD", y="Service", orientation="h", title="Total Cost by Service")
    st.plotly_chart(fig_svc, use_container_width=True)

# ------------------------------
# Tab 5: Tag Remediation Workflow (Task Set 5)
# ------------------------------
with tabs[4]:
    st.header("Task Set 5 – Tag Remediation Workflow")

    st.caption("Edit the cells below to simulate tagging fixes (Department, Project, Environment, Owner, CostCenter, CreatedBy, Tagged). Changes persist during this session.")

    editable_cols = ["Department","Project","Environment","Owner","CostCenter","CreatedBy","Tagged"]
    show_cols = ["ResourceID","Service","Region"] + editable_cols + ["MonthlyCostUSD","TagCompletenessScore"]

    edited = st.data_editor(
        st.session_state.df_remediated[show_cols],
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Tagged": st.column_config.SelectboxColumn(options=["Yes","No"], required=False),
        }
    )

    # Apply edits back to session df
    for col in editable_cols:
        st.session_state.df_remediated[col] = edited[col]

    # Recompute completeness after edits
    st.session_state.df_remediated["TagCompletenessScore"] = st.session_state.df_remediated.apply(tag_completeness_score, axis=1)

    st.subheader("5.4 Before vs After (Key Metrics)")
    before = tagging_metrics(apply_filters(st.session_state.df_original))
    after = tagging_metrics(apply_filters(st.session_state.df_remediated))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Untagged Resources (Before → After)", f"{after['n_untagged']:,}", delta=int(after["n_untagged"]-before["n_untagged"]))
    with c2:
        st.metric("Untagged Cost (Before → After)", f"${after['untagged_cost']:,.2f}", delta=after["untagged_cost"]-before["untagged_cost"])
    with c3:
        st.metric("% Cost Untagged (Before → After)", fmt_pct(after["pct_untagged_cost"]), delta=(after["pct_untagged_cost"]-before["pct_untagged_cost"]) if (pd.notna(after["pct_untagged_cost"]) and pd.notna(before["pct_untagged_cost"])) else None)

    st.subheader("5.5 Reflection (write here)")
    st.text_area(
        "How does improved tagging affect accountability and cost reports?",
        placeholder="E.g., With tags completed, costs are attributable to departments/projects, enabling showback/chargeback and driving better ownership...",
        height=120
    )

# ------------------------------
# Tab 6: Downloads & Short Report (Deliverables)
# ------------------------------
with tabs[5]:
    st.header("Deliverables – Downloads & Short Report")

    st.subheader("Datasets")
    st.download_button(
        "Download original.csv",
        data=st.session_state.df_original.to_csv(index=False).encode("utf-8"),
        file_name="original.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download remediated.csv",
        data=st.session_state.df_remediated.to_csv(index=False).encode("utf-8"),
        file_name="remediated.csv",
        mime="text/csv"
    )

    st.subheader("Short Report Summary")
    # Compute using FULL (unfiltered) for final deliverable clarity
    rep_before = tagging_metrics(st.session_state.df_original)
    rep_after = tagging_metrics(st.session_state.df_remediated)

    def block(label, before_val, after_val, fmt=lambda x: x):
        c1, c2, c3 = st.columns([2,2,2])
        c1.write(f"**{label}**")
        c2.write(f"Before: {fmt(before_val)}")
        c3.write(f"After:  {fmt(after_val)}")

    block("% of untagged resources", rep_before["pct_untagged"], rep_after["pct_untagged"], fmt_pct)
    block("Total untagged cost (USD)", rep_before["untagged_cost"], rep_after["untagged_cost"], lambda x: f"${x:,.2f}")
    # Departments with missing tags (count records missing any of the TAG_FIELDS)
    def dept_missing_counts(dfin):
        d = dfin.copy()
        miss_any = np.zeros(len(d), dtype=bool)
        for f in TAG_FIELDS:
            miss_any |= (d[f].isna() | (d[f].astype(str).str.strip() == ""))
        d["_MissingAnyTag"] = miss_any
        return (d[d["_MissingAnyTag"]]
                .groupby("Department", dropna=False)["ResourceID"]
                .count()
                .sort_values(ascending=False)
                .rename("MissingTagCount")
                .reset_index())

    dept_before = dept_missing_counts(st.session_state.df_original).head(10)
    dept_after = dept_missing_counts(st.session_state.df_remediated).head(10)

    st.write("**Departments with Missing Tags (Before)**")
    st.dataframe(dept_before, use_container_width=True)
    st.write("**Departments with Missing Tags (After)**")
    st.dataframe(dept_after, use_container_width=True)

    st.write("**Recommendations for Governance Improvement**")
    st.markdown(
        """
- Enforce a **required tag policy** (Department, Project, Environment, Owner, CostCenter, CreatedBy) at provisioning via IaC (e.g., Terraform `variables` + `validation`) and CI/CD checks.
- Add **Service Control Policies** or **Organization Tag Policies** to block launches missing required tags.
- Implement **weekly tag audits** with automated reports of untagged resources and owners (email/Slack).
- Apply **cost allocation tags** in the billing console; mandate **Owner** to enable chargeback/showback.
- Configure **lifecycle automation** (e.g., shut down or quarantine resources missing tags after grace period).
        """
    )

# Footer context
st.caption("Dataset model: CloudMart Inc. (simulated). App: Week 10 — Resource Tagging Cost Governance.")
