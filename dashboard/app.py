import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Predictive Sales Analytics",
    page_icon="📊",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "sales_processed.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "model_comparison.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

results = pd.read_csv(RESULTS_PATH)


# ==========================================
# TITLE
# ==========================================

st.title("📊 Predictive Sales Analytics")

st.markdown(
    """
    ### AI-powered Sales Prediction & Business Intelligence Dashboard

    Analyze historical sales performance and generate
    machine learning-based sales predictions.
    """
)


st.divider()


# ==========================================
# KPI CARDS
# ==========================================

total_sales = df["Sales"].sum()

total_profit = df["Profit"].sum()

total_records = len(df)

average_sales = df["Sales"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )


with col2:

    st.metric(
        "📈 Total Profit",
        f"${total_profit:,.2f}"
    )


with col3:

    st.metric(
        "📦 Records",
        f"{total_records:,}"
    )


with col4:

    st.metric(
        "💵 Average Sales",
        f"${average_sales:,.2f}"
    )


st.divider()


# ==========================================
# FILTERS
# ==========================================

st.sidebar.header("🔎 Filters")


selected_region = st.sidebar.multiselect(

    "Select Region",

    options=sorted(
        df["Region"].unique()
    ),

    default=sorted(
        df["Region"].unique()
    )
)


selected_category = st.sidebar.multiselect(

    "Select Category",

    options=sorted(
        df["Category"].unique()
    ),

    default=sorted(
        df["Category"].unique()
    )
)


filtered_df = df[

    df["Region"].isin(selected_region)

    &

    df["Category"].isin(selected_category)

]


# ==========================================
# SALES ANALYSIS
# ==========================================

st.header("📈 Sales Analysis")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Sales by Category")

    category_sales = (

        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)

    )

    st.bar_chart(category_sales)


with col2:

    st.subheader("Sales by Region")

    region_sales = (

        filtered_df
        .groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)

    )

    st.bar_chart(region_sales)


# ==========================================
# PROFIT ANALYSIS
# ==========================================

st.subheader("💰 Profit by Category")


profit_category = (

    filtered_df
    .groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)

)


st.bar_chart(profit_category)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.header("🤖 Machine Learning Model Performance")


st.dataframe(
    results,
    use_container_width=True
)


best_model = results.iloc[0]


st.success(

    f"🏆 Best Model: {best_model['Model']} | "
    f"R² Score: {best_model['R2']:.4f}"

)


# ==========================================
# SALES PREDICTION
# ==========================================

st.header("🔮 Predict Future Sales")


st.write(
    "Enter order details to generate a sales prediction."
)


col1, col2, col3 = st.columns(3)


with col1:

    ship_mode = st.selectbox(

        "Ship Mode",

        sorted(
            df["Ship Mode"].unique()
        )

    )

    segment = st.selectbox(

        "Segment",

        sorted(
            df["Segment"].unique()
        )

    )

    region = st.selectbox(

        "Region",

        sorted(
            df["Region"].unique()
        )

    )

    category = st.selectbox(

        "Category",

        sorted(
            df["Category"].unique()
        )

    )


with col2:

    city = st.selectbox(

        "City",

        sorted(
            df["City"].unique()
        )

    )

    state = st.selectbox(

        "State",

        sorted(
            df["State"].unique()
        )

    )

    country = st.selectbox(

        "Country",

        sorted(
            df["Country"].unique()
        )

    )

    sub_category = st.selectbox(

        "Sub-Category",

        sorted(
            df["Sub-Category"].unique()
        )

    )


with col3:

    postal_code = st.number_input(

        "Postal Code",

        min_value=0,

        value=10001

    )

    quantity = st.number_input(

        "Quantity",

        min_value=1,

        value=2

    )

    discount = st.slider(

        "Discount",

        min_value=0.0,

        max_value=0.8,

        value=0.1,

        step=0.05

    )


# ==========================================
# PREDICT
# ==========================================

if st.button(
    "🚀 Predict Sales",
    use_container_width=True
):

    input_data = pd.DataFrame([{

        "Ship Mode": ship_mode,

        "Segment": segment,

        "Country": country,

        "City": city,

        "State": state,

        "Postal Code": postal_code,

        "Region": region,

        "Category": category,

        "Sub-Category": sub_category,

        "Quantity": quantity,

        "Discount": discount

    }])


    prediction = model.predict(
        input_data
    )[0]


    st.success(
        f"💰 Predicted Sales: "
        f"${prediction:,.2f}"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Predictive Sales Analytics | "
    "Machine Learning + FastAPI + Streamlit"
)