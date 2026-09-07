"""
Customer Segmentation & Churn Pattern Analytics — European Banking
Streamlit dashboard.

Run locally with:
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    load_data, clean_and_segment, compute_kpis, segment_churn_table,
    train_model, score_customers, predict_single,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "European_Bank.csv")

st.set_page_config(
    page_title="European Bank — Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached data & model loading
# ---------------------------------------------------------------------

@st.cache_data
def get_data():
    raw = load_data(DATA_PATH)
    return clean_and_segment(raw)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return train_model(df)


@st.cache_data
def get_scored_data(_pipeline, df: pd.DataFrame):
    return score_customers(_pipeline, df)


full_df = get_data()
pipeline, diagnostics = get_model(full_df)
scored_df = get_scored_data(pipeline, full_df)


# ---------------------------------------------------------------------
# Sidebar — segment filters (apply across every tab)
# ---------------------------------------------------------------------

st.sidebar.title("🏦 Churn Analytics")
st.sidebar.caption("Customer Segmentation & Churn Pattern Analytics — France, Spain, Germany")
st.sidebar.markdown("---")
st.sidebar.subheader("Segment Filters")

geo_options = sorted(full_df["Geography"].unique().tolist())
geo_sel = st.sidebar.multiselect("Geography", geo_options, default=geo_options)

age_options = [g for g in full_df["AgeGroup"].cat.categories if g in full_df["AgeGroup"].unique()]
age_sel = st.sidebar.multiselect("Age Group", age_options, default=age_options)

tenure_options = list(full_df["TenureGroup"].cat.categories)
tenure_sel = st.sidebar.multiselect("Tenure Group", tenure_options, default=tenure_options)

balance_options = list(full_df["BalanceSegment"].cat.categories)
balance_sel = st.sidebar.multiselect("Balance Segment", balance_options, default=balance_options)

active_options = ["Active", "Inactive"]
active_sel = st.sidebar.multiselect("Activity Status", active_options, default=active_options)

st.sidebar.markdown("---")
st.sidebar.caption("Filters apply to every tab. Clear a filter box to exclude that segment.")

mask = (
    scored_df["Geography"].isin(geo_sel)
    & scored_df["AgeGroup"].isin(age_sel)
    & scored_df["TenureGroup"].isin(tenure_sel)
    & scored_df["BalanceSegment"].isin(balance_sel)
    & scored_df["IsActiveMember_Label"].isin(active_sel)
)
view_df = scored_df.loc[mask].copy()

if view_df.empty:
    st.warning("No customers match the current filter selection. Adjust the sidebar filters to see results.")
    st.stop()

kpis = compute_kpis(view_df)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab_overview, tab_geo, tab_hv, tab_model, tab_predict, tab_worklist = st.tabs([
    "📊 Overview", "🌍 Geography & Demographics", "💰 High-Value Explorer",
    "🤖 Model Insights", "🎯 Churn Risk Predictor", "📋 Retention Worklist",
])


# ---- Overview -------------------------------------------------------
with tab_overview:
    st.header("Overall Churn Summary")
    st.caption(f"Showing {len(view_df):,} of {len(full_df):,} customers based on current filters.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Churn Rate", f"{kpis['overall_churn_rate']:.1%}")
    c2.metric("Customers (filtered)", f"{kpis['n_customers']:,}")
    c3.metric("Churned Customers", f"{kpis['n_churned']:,}")
    c4.metric("Engagement Drop Indicator", f"{kpis['engagement_drop_pp']:.1f} pp",
              help="Churn rate of inactive members minus active members")

    c5, c6 = st.columns(2)
    c5.metric("High-Value Churn Ratio", f"{kpis['high_value_churn_ratio']:.1%}",
              help="Share of all churners who are high-value customers")
    c6.metric("Balance Lost to Churn", f"{kpis['churned_balance_share']:.1%}",
              help="Share of total account balance held by churned customers")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Rate by Segment Dimension")
        seg_choice = st.selectbox(
            "Segment dimension", ["Geography", "AgeGroup", "CreditScoreBand", "TenureGroup", "BalanceSegment"],
            key="overview_seg",
        )
        tbl = segment_churn_table(view_df, seg_choice).reset_index()
        fig = px.bar(tbl, x=seg_choice, y="ChurnRate", text_auto=".1%",
                     color="ChurnRate", color_continuous_scale="Reds")
        fig.add_hline(y=kpis["overall_churn_rate"], line_dash="dash", line_color="gray",
                      annotation_text="Overall rate")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Retained vs Churned Split")
        pie_df = view_df["Exited_Label"].value_counts().reset_index()
        pie_df.columns = ["Status", "Customers"]
        fig2 = px.pie(pie_df, names="Status", values="Customers", hole=0.45,
                      color="Status", color_discrete_map={"Retained": "#4C72B0", "Churned": "#C44E52"})
        st.plotly_chart(fig2, width='stretch')


# ---- Geography & Demographics ---------------------------------------
with tab_geo:
    st.header("Geography & Demographic Churn Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Rate by Geography")
        geo_tbl = segment_churn_table(view_df, "Geography").reset_index()
        fig = px.bar(geo_tbl, x="Geography", y="ChurnRate", text_auto=".1%",
                     color="Geography")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Churn Rate by Gender")
        gender_tbl = segment_churn_table(view_df, "Gender").reset_index()
        fig = px.bar(gender_tbl, x="Gender", y="ChurnRate", text_auto=".1%", color="Gender")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Geography × Age Group Interaction")
    geo_age = view_df.groupby(["Geography", "AgeGroup"], observed=True)["Exited"].mean().unstack()
    fig = px.imshow(geo_age, text_auto=".1%", color_continuous_scale="Reds",
                     labels=dict(color="Churn Rate"))
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Age & Tenure Churn Comparison")
    col3, col4 = st.columns(2)
    with col3:
        age_tbl = segment_churn_table(view_df, "AgeGroup").reset_index()
        fig = px.bar(age_tbl, x="AgeGroup", y="ChurnRate", text_auto=".1%", color="AgeGroup")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, title="Churn Rate by Age Group")
        st.plotly_chart(fig, width='stretch')
    with col4:
        tenure_tbl = segment_churn_table(view_df, "TenureGroup").reset_index()
        fig = px.bar(tenure_tbl, x="TenureGroup", y="ChurnRate", text_auto=".1%", color="TenureGroup")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, title="Churn Rate by Tenure Group")
        st.plotly_chart(fig, width='stretch')


# ---- High-Value Explorer --------------------------------------------
with tab_hv:
    st.header("High-Value Customer Churn Explorer")
    st.caption("High-value = top quartile by Balance or by Estimated Salary (computed on the full customer base).")

    hv_df = view_df[view_df["HighValue"]]
    if hv_df.empty:
        st.info("No high-value customers in the current filter selection.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("High-Value Customers", f"{len(hv_df):,}")
        c2.metric("High-Value Churn Rate", f"{hv_df['Exited'].mean():.1%}")
        c3.metric("Share of All Churners", f"{kpis['high_value_churn_ratio']:.1%}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("High-Value Churn by Geography")
            hv_geo = segment_churn_table(hv_df, "Geography").reset_index()
            fig = px.bar(hv_geo, x="Geography", y="ChurnRate", text_auto=".1%", color="Geography")
            fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Salary vs. Balance — Retained vs Churned")
            fig = px.scatter(view_df, x="EstimatedSalary", y="Balance", color="Exited_Label",
                              opacity=0.4, color_discrete_map={"Retained": "#4C72B0", "Churned": "#C44E52"})
            st.plotly_chart(fig, width='stretch')

        st.subheader("High-Value Customer Detail")
        st.dataframe(
            hv_df[["CustomerId", "Geography", "Age", "Balance", "EstimatedSalary",
                   "NumOfProducts", "IsActiveMember_Label", "ChurnProbability", "Exited_Label"]]
            .sort_values("ChurnProbability", ascending=False),
            width='stretch', height=350,
        )


# ---- Model Insights ---------------------------------------------------
with tab_model:
    st.header("Churn Prediction Model — Performance & Drivers")
    st.caption("Tuned XGBoost classifier (selected via cross-validated hyperparameter search in the analysis notebook).")

    m = diagnostics["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['Accuracy']:.1%}")
    c2.metric("Precision", f"{m['Precision']:.1%}")
    c3.metric("Recall", f"{m['Recall']:.1%}")
    c4.metric("F1 Score", f"{m['F1']:.3f}")
    c5.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    st.caption(f"Evaluated on a held-out test set of {diagnostics['n_test']:,} customers "
               f"(trained on {diagnostics['n_train']:,}).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = diagnostics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Retained", "Churned"], y=["Retained", "Churned"],
                         color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("ROC Curve")
        fpr, tpr = diagnostics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC={m['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Top Feature Importances")
    fi_df = diagnostics["feature_importance"].head(10)
    fig = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Greens")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')


# ---- Churn Risk Predictor (What-If) ----------------------------------
with tab_predict:
    st.header("What-If Churn Risk Predictor")
    st.caption("Enter a hypothetical customer profile to see the model's predicted churn probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        geography = st.selectbox("Geography", sorted(full_df["Geography"].unique().tolist()))
        gender = st.selectbox("Gender", sorted(full_df["Gender"].unique().tolist()))
        age = st.slider("Age", 18, 92, 40)
    with col2:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    with col3:
        balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        salary = st.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Active Member?", ["Yes", "No"])

    inputs = {
        "CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "EstimatedSalary": salary,
        "Geography": geography, "Gender": gender,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
    }

    if st.button("Predict Churn Risk", type="primary"):
        prob = predict_single(pipeline, inputs)
        tier = "High Risk" if prob >= 0.7 else ("Medium Risk" if prob >= 0.4 else "Low Risk")
        tier_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}[tier]

        st.markdown("### Result")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Churn Probability", f"{prob:.1%}")
        c2.metric("Risk Tier", f"{tier_color} {tier}")
        st.progress(min(max(prob, 0.0), 1.0))


# ---- Retention Worklist ----------------------------------------------
with tab_worklist:
    st.header("Prioritized Retention Worklist")
    st.caption("Customers ranked by predicted churn probability — a starting point for targeted retention outreach.")

    tier_filter = st.multiselect("Risk Tier", ["High Risk", "Medium Risk", "Low Risk"],
                                  default=["High Risk", "Medium Risk"])
    worklist = view_df[view_df["ChurnRiskTier"].isin(tier_filter)].sort_values(
        "ChurnProbability", ascending=False
    )[["CustomerId", "Geography", "Age", "Tenure", "Balance", "NumOfProducts",
       "IsActiveMember_Label", "ChurnProbability", "ChurnRiskTier", "Exited_Label"]]

    st.dataframe(worklist, width='stretch', height=450)
    st.download_button(
        "Download Worklist as CSV",
        data=worklist.to_csv(index=False).encode("utf-8"),
        file_name="churn_retention_worklist.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Customer Segmentation & Churn Pattern Analytics — European Banking | Built with Streamlit")
