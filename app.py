
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Ops + CX Dashboard", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def make_demo_data(seed: int = 7, n_stores: int = 120, days: int = 180):
    rng = np.random.default_rng(seed)
    # Stores
    franchises = ["Frisch's Big Boy", "Nathan's Famous", "Friendly's", "Dai Famo Pasta", "Burgero"]
    platforms = ["DoorDash", "UberEats", "Grubhub"]
    stores = []
    for i in range(n_stores):
        f = rng.choice(franchises, p=[0.36, 0.28, 0.16, 0.12, 0.08])
        city = rng.choice(["Cincinnati, OH","Detroit, MI","Columbus, OH","Lexington, KY","Pittsburgh, PA",
                           "Cleveland, OH","Indianapolis, IN","Dayton, OH","Toledo, OH","Louisville, KY"])
        stores.append({
            "store_id": 10000+i,
            "store_name": f"{f} #{100+i} ({city})",
            "franchise": f,
            "city": city
        })
    stores = pd.DataFrame(stores)

    # Daily revenue facts
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days-1)
    date_index = pd.date_range(start, periods=days, freq="D")

    rows = []
    for _, s in stores.iterrows():
        # base performance by franchise
        base = {
            "Frisch's Big Boy": 280.0,
            "Nathan's Famous": 60.0,
            "Friendly's": 140.0,
            "Dai Famo Pasta": 95.0,
            "Burgero": 120.0
        }[s.franchise]
        vol = base * (1 + rng.normal(0, 0.15))
        for d in date_index:
            dow = d.dayofweek # 0=Mon
            # day-of-week effect
            vol_dow = vol * (1.00 + [ -0.05, -0.02, 0.00, 0.06, 0.12, 0.10, 0.08 ][dow])
            # monthly effect
            vol_month = vol_dow * (1 + 0.15*np.sin((d.dayofyear/365)*2*np.pi))
            # noise
            x = max(0, vol_month + rng.normal(0, vol*0.25))
            orders = max(0, int(rng.normal(200 if vol>200 else 110, 25)))
            aov = np.clip(rng.normal( (x / max(1,orders))*1.05, 2.0), 6, 25)
            refunds = np.clip(int(rng.normal(orders * (0.05 if s.franchise!="Nathan's Famous" else 0.15), 3)), 0, orders)
            refund_amt = np.clip(refunds * rng.normal(6.0, 1.25), 0, None)
            platform = np.random.choice(platforms, p=[0.55, 0.30, 0.15])
            rows.append({
                "date": d, "store_id": s.store_id, "store_name": s.store_name, "franchise": s.franchise, "city": s.city,
                "orders": orders, "avg_order_value": aov, "total_revenue": x, "platform": platform,
                "refunds": refunds, "refund_amount": refund_amt
            })
    facts = pd.DataFrame(rows)
    facts["refund_rate"] = (facts["refunds"] / facts["orders"]).replace([np.inf, np.nan], 0.0)

    # Ratings (simulate significant skew to 1-star)
    rating_rows = []
    for _, s in stores.iterrows():
        n = int(np.clip(abs(rng.normal(320, 120)), 40, 1200))
        # 1-star heavy
        weights = np.array([0.80, 0.02, 0.03, 0.04, 0.11])
        stars = rng.choice([1,2,3,4,5], size=n, p=weights/weights.sum())
        mean = stars.mean()
        # Bayesian shrinkage toward global 2.2★ with min 20 reviews
        global_mean = 2.2
        prior_n = 20
        adj = (stars.sum() + prior_n*global_mean) / (n + prior_n)
        rating_rows.append({
            "store_id": s.store_id, "store_name": s.store_name, "franchise": s.franchise,
            "reviews": n, "mean_star": mean, "adj_star": adj
        })
    ratings = pd.DataFrame(rating_rows)
    return facts, ratings, stores

def maybe_use_uploaded_data():
    st.sidebar.markdown("### Upload your CSVs (optional)")
    revenue_csv = st.sidebar.file_uploader(
        "Operational facts (daily grain). Columns required: date, store_id, store_name, franchise, city, orders, avg_order_value, total_revenue, refunds, refund_amount, platform",
        type=["csv"],
        key="rev")
    ratings_csv = st.sidebar.file_uploader(
        "Ratings. Columns required: store_id, store_name, franchise, reviews, mean_star, adj_star",
        type=["csv"],
        key="rat")
    if revenue_csv is not None:
        facts = pd.read_csv(revenue_csv, parse_dates=["date"])
        facts["refund_rate"] = (facts["refunds"] / facts["orders"]).replace([np.inf, np.nan], 0.0)
    else:
        facts = None
    ratings = pd.read_csv(ratings_csv) if ratings_csv is not None else None
    return facts, ratings

# ------------------------------
# Data
# ------------------------------
facts_demo, ratings_demo, stores = make_demo_data()
facts_up, ratings_up = maybe_use_uploaded_data()
facts = facts_up if facts_up is not None else facts_demo
ratings = ratings_up if ratings_up is not None else ratings_demo

min_date, max_date = facts["date"].min(), facts["date"].max()

# ------------------------------
# Sidebar filters
# ------------------------------
st.sidebar.markdown("## Filters")
date_range = st.sidebar.date_input("Date range", [min_date.date(), max_date.date()],
                                   min_value=min_date.date(), max_value=max_date.date())
start_date = pd.Timestamp(date_range[0])
end_date = pd.Timestamp(date_range[-1]) + pd.Timedelta(days=1)

franchises = ["All"] + sorted(facts["franchise"].unique().tolist())
franchise_sel = st.sidebar.selectbox("Franchise", franchises)
store_sel = st.sidebar.multiselect("Store (optional)", sorted(facts["store_name"].unique().tolist()))

metric_sel = st.sidebar.selectbox("Metric", ["total_revenue","orders","avg_order_value","refund_rate"])

f = facts.query("@start_date <= date < @end_date").copy()
if franchise_sel != "All":
    f = f[f["franchise"]==franchise_sel]
if store_sel:
    f = f[f["store_name"].isin(store_sel)]

# ------------------------------
# KPIs
# ------------------------------
def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

col1, col2, col3, col4 = st.columns(4)
kpi_card("Total Revenue ($)", f"${f['total_revenue'].sum():,.0f}")
kpi_card("Median Daily Revenue", f"${f.groupby('date')['total_revenue'].sum().median():,.0f}")
kpi_card("Refund Rate", f"{100*f['refunds'].sum()/max(1,f['orders'].sum()):.1f}%")
kpi_card("Avg Rating (adj)", f"{ratings['adj_star'].mean():.2f}★", help_text="Bayesian adjusted star rating")

st.markdown("---")

# ------------------------------
# Tabs
# ------------------------------
tab_overview, tab_rev, tab_model, tab_cx, tab_actions = st.tabs([
    "Overview", "Revenue Analysis", "Model & Prediction", "Customer Satisfaction", "Actionable Insights"
])

# -------------- Overview
with tab_overview:
    st.subheader("Top & Bottom Stores")
    g = f.groupby(["store_id","store_name"])["total_revenue"].sum().reset_index()
    topN = g.nlargest(20, "total_revenue")
    botN = g.nsmallest(10, "total_revenue")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(topN.sort_values("total_revenue"),
                     x="total_revenue", y="store_name", orientation="h",
                     title="Top 20 Best Performing Stores",
                     labels={"total_revenue":"Total Revenue ($)","store_name":""})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # color by refund rate severity
        ref = f.groupby(["store_id","store_name"]).agg(
            total_rev=("total_revenue","sum"),
            refund_rate=("refunds",lambda x: x.sum()/max(1, f.loc[x.index,"orders"].sum()))
        ).reset_index()
        bot = ref.nsmallest(10,"total_rev").sort_values("total_rev")
        fig2 = px.bar(bot, x="total_rev", y="store_name", orientation="h",
                      color="refund_rate", color_continuous_scale="Reds",
                      title="Bottom 10 Stores (color = refund rate)",
                      labels={"total_rev":"Total Revenue ($)","store_name":""})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Revenue Distribution")
    daily_store = f.groupby(["store_id","date"])["total_revenue"].sum().reset_index()
    fig3 = px.histogram(daily_store, x="total_revenue", nbins=50,
                        title="Distribution of Store Daily Revenue",
                        labels={"total_revenue":"Total Revenue ($)"})
    fig3.add_vline(x=daily_store["total_revenue"].median(), line_dash="dash", annotation_text="Median")
    st.plotly_chart(fig3, use_container_width=True)

# -------------- Revenue
with tab_rev:
    st.subheader("Revenue by Day / Month / Quarter")
    # Day of week
    df_dow = f.copy()
    df_dow["dow"] = df_dow["date"].dt.day_name()
    rev_dow = df_dow.groupby("dow")["total_revenue"].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).reset_index()
    fig_dow = px.bar(rev_dow, x="dow", y="total_revenue",
                     title="Total Revenue by Day of Week", labels={"total_revenue":"Total Revenue ($)","dow":""})
    st.plotly_chart(fig_dow, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        df_m = f.copy()
        df_m["month"] = df_m["date"].dt.month_name()
        order_m = list(pd.date_range(f["date"].min(), f["date"].max(), freq="MS").month_name().unique())
        rev_m = df_m.groupby("month")["total_revenue"].sum().reindex(order_m).reset_index()
        fig_m = px.bar(rev_m, x="month", y="total_revenue",
                       title="Total Revenue by Month", labels={"total_revenue":"Total Revenue ($)","month":""})
        st.plotly_chart(fig_m, use_container_width=True)
    with c2:
        df_q = f.copy()
        df_q["q"] = "Q" + df_q["date"].dt.quarter.astype(str)
        rev_q = df_q.groupby("q")["total_revenue"].sum().reset_index()
        fig_q = px.bar(rev_q, x="q", y="total_revenue", text_auto=".0f",
                       title="Total Revenue by Quarter", labels={"total_revenue":"Total Revenue ($)","q":""})
        st.plotly_chart(fig_q, use_container_width=True)

    st.subheader("Orders vs Avg Revenue (bubble size = refund rate)")
    agg = f.groupby("store_id").agg(
        total_orders=("orders","sum"),
        avg_revenue=("total_revenue",lambda x: x.sum()/max(1, len(x))),
        refund_rate=("refunds",lambda x: x.sum()/max(1, f.loc[x.index,"orders"].sum()))
    ).reset_index().merge(stores, on="store_id")
    fig_bub = px.scatter(agg, x="total_orders", y="avg_revenue",
                         size="refund_rate", color="avg_revenue",
                         hover_data=["store_name","franchise","city"],
                         labels={"total_orders":"Total Orders","avg_revenue":"Average Daily Revenue ($)"},
                         title="Orders vs Average Daily Revenue")
    st.plotly_chart(fig_bub, use_container_width=True)

# -------------- Model & Prediction (simulated)
with tab_model:
    st.caption("Demo section: import your own model outputs to replace these visuals.")
    st.subheader("Top Revenue Drivers (Feature Importance)")
    # Demo importances
    imp = pd.DataFrame({
        "feature":["avg_order_value","avg_subtotal","order_count","doordash_orders","num_stores","rain_days","marketing_spend","north_region","promos","avg_prep_time"],
        "importance":[0.47,0.29,0.09,0.06,0.05,0.02,0.01,0.005,0.003,0.002]
    })
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Top 10 Feature Importances")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Model Prediction Accuracy")
    # Simulate actual vs predicted
    days = pd.date_range(f["date"].min(), f["date"].max(), freq="D")
    actual = f.groupby("date")["total_revenue"].sum().reindex(days).fillna(method="ffill")
    noise_linear = np.random.normal(0, 16.87, size=len(days))
    noise_rf = np.random.normal(0, 3.21, size=len(days))
    pred_lr = actual + noise_linear
    pred_rf = actual + noise_rf

    df_lr = pd.DataFrame({"model":"Linear Regression", "actual":actual.values, "predicted":pred_lr})
    df_rf = pd.DataFrame({"model":"Random Forest", "actual":actual.values, "predicted":pred_rf})
    dfp = pd.concat([df_lr, df_rf], ignore_index=True)

    fig_sc = px.scatter(dfp, x="actual", y="predicted", color="model",
                        opacity=0.65, trendline="ols",
                        labels={"actual":"Actual Revenue ($)","predicted":"Predicted Revenue ($)"},
                        title="Predicted vs Actual")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Residuals (Actual - Predicted)")
    dfp["residual"] = dfp["actual"] - dfp["predicted"]
    fig_res = px.scatter(dfp, x="predicted", y="residual", color="model",
                         labels={"predicted":"Predicted Revenue ($)","residual":"Residual ($)"},
                         title="Residual Plot")
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

# -------------- Customer Satisfaction (R)
with tab_cx:
    st.subheader("Adjusted Star Ratings (Bayesian)")
    top15 = ratings.nlargest(15, "adj_star")
    bot15 = ratings.nsmallest(15, "adj_star")

    c1, c2 = st.columns(2)
    with c1:
        fig_top = px.bar(top15.sort_values("adj_star"), x="adj_star", y="store_name", orientation="h",
                         title="Top 15 Stores by Adjusted Star Rating (adj★)",
                         labels={"adj_star":"Adjusted Star (★)","store_name":""})
        st.plotly_chart(fig_top, use_container_width=True)
    with c2:
        fig_bot = px.bar(bot15.sort_values("adj_star"), x="adj_star", y="store_name", orientation="h",
                         title="Bottom 15 Stores by Adjusted Star Rating (adj★)",
                         labels={"adj_star":"Adjusted Star (★)","store_name":""})
        st.plotly_chart(fig_bot, use_container_width=True)

    st.subheader("Distribution of Ratings")
    fig_hist = px.histogram(ratings, x="mean_star", nbins=5, histnorm=None,
                            title="Review Counts by Star (mean per store)",
                            labels={"mean_star":"Star rating"})
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------- Actionable Insights
with tab_actions:
    st.subheader("Highlights & Alerts")
    # Simple rules for demo
    overall_refund_rate = f["refunds"].sum() / max(1, f["orders"].sum())
    high_refund_franchises = (f.groupby("franchise")
                                .apply(lambda d: d["refunds"].sum()/max(1,d["orders"].sum()))
                                .sort_values(ascending=False))
    st.write(f"**Portfolio refund rate:** {overall_refund_rate:.1%}")
    worst = high_refund_franchises.index[0]
    st.write(f"**Highest refund rate franchise:** {worst} — {high_refund_franchises.iloc[0]:.1%}")
    st.markdown(\"\"\"
- Focus on stores with **high refunds and high orders** (bottom-left of bubble chart).
- Check **bag-check discipline**, **pickup lanes**, and **peak staffing** where adjusted stars are lowest.
- Use **AOV** and **subtotal** levers (top features) to drive revenue; volume alone is not sufficient.
- Random Forest shows **stable, low-error predictions** — better for near‑term forecasting.
    \"\"\")
