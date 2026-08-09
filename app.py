from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Restaurant Food Recommendation Engine")

MODEL_PATH = "trained_models/recommendation_model.pkl"
DATASET_PATH = "dataset/restaurant_orders.csv"

model_data = None


def load_model():
    global model_data
    model_data = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    load_model()


class OrderRequest(BaseModel):
    customer_id: int
    age_group: str
    gender: str
    visit_frequency: str
    preferred_cuisine: str
    favorite_food_category: str
    previous_orders: int
    frequently_ordered_item: str
    ordered_item: str
    average_bill_amount: float
    time_of_visit: str
    day_of_week: str
    season: str
    veg_nonveg_pref: str
    spice_preference: str
    customer_rating: float
    ordered_again: int


def get_popular_dishes(top_n=5):
    df = model_data["df"]
    dish_stats = df.groupby("ordered_item").agg(
        avg_rating=("customer_rating", "mean"),
        order_count=("ordered_item", "count")
    ).reset_index()

    dish_stats["popularity_score"] = (
        (dish_stats["avg_rating"] / 5) * 0.6 +
        (dish_stats["order_count"] / dish_stats["order_count"].max()) * 0.4
    )
    top_dishes = dish_stats.sort_values("popularity_score", ascending=False).head(top_n)

    results = []
    max_score = top_dishes["popularity_score"].max()
    for _, row in top_dishes.iterrows():
        confidence = round(float(min(row["popularity_score"] / max_score * 100, 99.9)), 1)
        results.append({
            "food": row["ordered_item"],
            "confidence": confidence,
            "reason": "Popular and highly rated among all customers"
        })
    return results


def get_trending_dishes(top_n=5, days=30):
    df = model_data["df"].copy()
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    cutoff = df["order_timestamp"].max() - pd.Timedelta(days=days)
    recent = df[df["order_timestamp"] >= cutoff]

    if recent.empty:
        return []

    trend_counts = recent["ordered_item"].value_counts().head(top_n)
    max_count = trend_counts.max()

    results = []
    for food, count in trend_counts.items():
        confidence = round(float(min(count / max_count * 100, 99.9)), 1)
        results.append({
            "food": food,
            "confidence": confidence,
            "reason": f"Trending — ordered {count} times in the last {days} days"
        })
    return results


def get_seasonal_dishes(season=None, top_n=5):
    df = model_data["df"]
    if season is None:
        season = df["season"].mode()[0]

    seasonal_orders = df[df["season"] == season]
    if seasonal_orders.empty:
        return []

    top_dishes = seasonal_orders["ordered_item"].value_counts().head(top_n)
    max_count = top_dishes.max()

    results = []
    for food, count in top_dishes.items():
        confidence = round(float(min(count / max_count * 100, 99.9)), 1)
        results.append({
            "food": food,
            "confidence": confidence,
            "reason": f"Popular during {season}"
        })
    return results


def get_combo_suggestion(dish, top_n=1):
    df = model_data["df"]
    dish_row_category = df[df["ordered_item"] == dish]["favorite_food_category"].mode()
    if dish_row_category.empty:
        return None
    dish_category = dish_row_category[0]

    customers_who_ordered = df[df["ordered_item"] == dish]["customer_id"].unique()

    other_orders = df[
        (df["customer_id"].isin(customers_who_ordered)) &
        (df["ordered_item"] != dish) &
        (df["favorite_food_category"] != dish_category)
    ]

    if other_orders.empty:
        return None

    combo = other_orders["ordered_item"].value_counts().head(top_n)
    return combo.index[0] if not combo.empty else None


def customers_also_ordered(dish, top_n=5):
    df = model_data["df"]
    customers_who_ordered = df[df["ordered_item"] == dish]["customer_id"].unique()

    other_orders = df[
        (df["customer_id"].isin(customers_who_ordered)) &
        (df["ordered_item"] != dish)
    ]

    if other_orders.empty:
        return {"dish": dish, "customers_also_ordered": []}

    top_dishes = other_orders["ordered_item"].value_counts().head(top_n)
    max_count = top_dishes.max()

    results = []
    for food, count in top_dishes.items():
        confidence = round(float(min(count / max_count * 100, 99.9)), 1)
        results.append({
            "food": food,
            "confidence": confidence,
            "co_order_count": int(count)
        })

    return {"dish": dish, "customers_also_ordered": results}


def recommend_dishes(customer_id, top_n=5):
    df = model_data["df"]
    customer_profile = model_data["customer_profile"]
    customer_sim_df = model_data["customer_sim_df"]

    if customer_id not in customer_profile["customer_id"].values:
        return {
            "customer_id": int(customer_id),
            "recommendations": get_popular_dishes(top_n),
            "note": "New customer detected — showing popular picks until order history builds up"
        }

    cust_row = customer_profile[customer_profile["customer_id"] == customer_id].iloc[0]
    cust_orders = df[df["customer_id"] == customer_id]

    cust_common_time = cust_orders["time_of_visit"].mode()[0] if not cust_orders.empty else None
    cust_common_day = cust_orders["day_of_week"].mode()[0] if not cust_orders.empty else None
    cust_common_season = cust_orders["season"].mode()[0] if not cust_orders.empty else None

    all_dishes = df["ordered_item"].unique()
    dish_scores = {}

    for dish in all_dishes:
        dish_rows = df[df["ordered_item"] == dish]
        content_score = 0

        if (dish_rows["preferred_cuisine"] == cust_row["preferred_cuisine"]).mean() > 0.3:
            content_score += 0.30
        if (dish_rows["favorite_food_category"] == cust_row["favorite_food_category"]).any():
            content_score += 0.15
        if (dish_rows["veg_nonveg_pref"] == cust_row["veg_nonveg_pref"]).mean() > 0.5:
            content_score += 0.10
        if (dish_rows["spice_preference"] == cust_row["spice_preference"]).mean() > 0.3:
            content_score += 0.10

        collab_score = 0
        orderers = dish_rows["customer_id"].unique()
        sims = [customer_sim_df.loc[customer_id, o] for o in orderers
                if o != customer_id and o in customer_sim_df.columns]
        if sims:
            collab_score = np.mean(sims) * 0.25

        avg_rating = dish_rows["customer_rating"].mean()
        rating_score = (avg_rating / 5) * 0.15

        already_ordered = dish in cust_orders["ordered_item"].values
        recency_score = 0.10 if already_ordered and cust_orders["ordered_again"].mean() > 0.5 else 0

        time_score = 0
        if cust_common_time and (dish_rows["time_of_visit"] == cust_common_time).mean() > 0.3:
            time_score = 0.05

        day_score = 0
        if cust_common_day and (dish_rows["day_of_week"] == cust_common_day).mean() > 0.2:
            day_score = 0.03

        season_score = 0
        if cust_common_season and (dish_rows["season"] == cust_common_season).mean() > 0.3:
            season_score = 0.02

        total = (content_score + collab_score + rating_score + recency_score
                  + time_score + day_score + season_score)

        dish_scores[dish] = {
            "score": total,
            "content_score": content_score,
            "collab_score": collab_score,
            "rating_score": rating_score,
            "recency_score": recency_score,
            "time_score": time_score,
            "day_score": day_score,
            "season_score": season_score,
            "avg_rating": round(avg_rating, 2)
        }

    ranked = sorted(dish_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]

    results = []
    max_score = max(d[1]["score"] for d in ranked) if ranked else 1

    for dish, s in ranked:
        confidence = round(float(min(s["score"] / max_score * 100, 99.9)), 1)

        signal_strengths = {
            "Frequently ordered by you before": s["recency_score"] / 0.10,
            "Highly rated by customers with similar preferences": s["rating_score"] / 0.15,
            "Customers with similar taste preferred this": s["collab_score"] / 0.25,
            "Matches your preferred cuisine": s["content_score"] / 0.65,
            "Popular during your usual visit time": s["time_score"] / 0.05 if s["time_score"] else 0,
            "Popular on your usual visit day": s["day_score"] / 0.03 if s["day_score"] else 0,
            "Matches seasonal preferences": s["season_score"] / 0.02 if s["season_score"] else 0,
        }
        best_reason = max(signal_strengths, key=signal_strengths.get)
        if signal_strengths[best_reason] == 0:
            best_reason = "Complements your recent orders"

        combo = get_combo_suggestion(dish)

        rec = {
            "food": dish,
            "confidence": confidence,
            "reason": best_reason
        }
        if combo:
            rec["combo_suggestion"] = combo

        results.append(rec)

    return {"customer_id": int(customer_id), "recommendations": results}


def rebuild_model():
    df = pd.read_csv(DATASET_PATH)

    customer_profile = df.groupby("customer_id").agg({
        "preferred_cuisine": lambda x: x.mode()[0],
        "favorite_food_category": lambda x: x.mode()[0],
        "veg_nonveg_pref": lambda x: x.mode()[0],
        "spice_preference": lambda x: x.mode()[0],
        "visit_frequency": lambda x: x.mode()[0],
        "customer_rating": "mean",
        "average_bill_amount": "mean"
    }).reset_index()

    profile_encoded = pd.get_dummies(
        customer_profile[["preferred_cuisine", "favorite_food_category",
                           "veg_nonveg_pref", "spice_preference", "visit_frequency"]]
    )
    profile_encoded["customer_rating"] = customer_profile["customer_rating"]
    profile_encoded["average_bill_amount"] = customer_profile["average_bill_amount"]

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(profile_encoded)
    customer_sim_df = pd.DataFrame(similarity,
                                    index=customer_profile["customer_id"],
                                    columns=customer_profile["customer_id"])

    new_model_data = {
        "customer_profile": customer_profile,
        "customer_sim_df": customer_sim_df,
        "df": df
    }
    joblib.dump(new_model_data, MODEL_PATH)
    return new_model_data


@app.get("/")
def root():
    return {"message": "Restaurant Recommendation Engine API is running"}


@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: int):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return recommend_dishes(customer_id, top_n=5)


@app.get("/trending")
def get_trending(days: int = 30, top_n: int = 5):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"trending_dishes": get_trending_dishes(top_n=top_n, days=days)}


@app.get("/seasonal")
def get_seasonal(season: str = None, top_n: int = 5):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"season": season or "current", "seasonal_dishes": get_seasonal_dishes(season, top_n)}


@app.get("/also-ordered/{dish_name}")
def get_also_ordered(dish_name: str, top_n: int = 5):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return customers_also_ordered(dish_name, top_n=top_n)


@app.post("/order")
def add_order(order: OrderRequest):
    try:
        df = pd.read_csv(DATASET_PATH)
        new_row = order.dict()
        new_row["order_timestamp"] = datetime.now().isoformat()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATASET_PATH, index=False)
        return {"message": "Order recorded successfully", "customer_id": order.customer_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def retrain_model():
    global model_data
    try:
        model_data = rebuild_model()
        return {"message": "Model retrained successfully",
                "customers": int(model_data["customer_profile"].shape[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)