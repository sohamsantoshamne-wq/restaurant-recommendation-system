import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Restaurant Recommendation Analytics", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/restaurant_orders.csv")

df = load_data()

st.title("🍽️ Restaurant Recommendation Engine — Analytics Dashboard")

# ---- Top metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", df["customer_id"].nunique())
col2.metric("Total Orders", len(df))
col3.metric("Avg Bill Amount", f"₹{df['average_bill_amount'].mean():.0f}")
col4.metric("Avg Rating", f"{df['customer_rating'].mean():.2f} ⭐")

st.divider()

# ---- Most ordered items ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Most Ordered Items")
    top_items = df["ordered_item"].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_items.values, y=top_items.index, ax=ax, palette="viridis")
    ax.set_xlabel("Orders")
    st.pyplot(fig)

with col2:
    st.subheader("Cuisine Popularity")
    cuisine_counts = df["preferred_cuisine"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cuisine_counts.values, labels=cuisine_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Visit Distribution")
    visit_counts = df["visit_frequency"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=visit_counts.index, y=visit_counts.values, ax=ax, palette="magma")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

with col2:
    st.subheader("Category Distribution")
    cat_counts = df["favorite_food_category"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax, palette="coolwarm")
    ax.set_ylabel("Orders")
    plt.xticks(rotation=30)
    st.pyplot(fig)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Distribution (Synthetic)")
    fig, ax = plt.subplots()
    sns.histplot(df["average_bill_amount"], bins=30, kde=True, ax=ax, color="teal")
    ax.set_xlabel("Bill Amount (₹)")
    st.pyplot(fig)

with col2:
    st.subheader("Time of Visit Breakdown")
    time_counts = df["time_of_visit"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=time_counts.index, y=time_counts.values, ax=ax, palette="crest")
    ax.set_ylabel("Orders")
    st.pyplot(fig)

st.divider()
st.subheader("🔥 Trending Foods (Last 30 Days)")

df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
recent_cutoff = df["order_timestamp"].max() - pd.Timedelta(days=30)
recent_orders = df[df["order_timestamp"] >= recent_cutoff]

if not recent_orders.empty:
    trending = recent_orders["ordered_item"].value_counts().head(10)
    st.bar_chart(trending)
else:
    st.info("Not enough recent order data to show trends.")

st.divider()
st.subheader("Recommendation Accuracy (from Model Evaluation)")
st.info("Precision@5: 0.4874  |  Recall@5: 0.8369  |  F1 Score: 0.6160  |  Accuracy: 0.7482  (evaluated on 190 customers)")