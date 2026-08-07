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


# ---------- Request Schemas ----------
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


# ---------- Core recommendation logic (same as notebook) ----------
def recommend_dishes(customer_id, top_n=5):
    df = model_data["df"]
    customer_profile = model_data["customer_profile"]
    customer_sim_df = model_data["customer_sim_df"]

    if customer_id not in customer_profile["customer_id"].values:
        return {"error": "Customer not found"}

    cust_row = customer_profile[customer_profile["customer_id"] == customer_id].iloc[0]
    cust_orders = df[df["customer_id"] == customer_id]

    all_dishes = df["ordered_item"].unique()
    dish_scores = {}

    for dish in all_dishes:
        dish_rows = df[df["ordered_item"] == dish]
        content_score = 0

        if (dish_rows["preferred_cuisine"] == cust_row["preferred_cuisine"]).mean() > 0.3:
            content_score += 0.35
        if (dish_rows["favorite_food_category"] == cust_row["favorite_food_category"]).any():
            content_score += 0.2
        if (dish_rows["veg_nonveg_pref"] == cust_row["veg_nonveg_pref"]).mean() > 0.5:
            content_score += 0.15
        if (dish_rows["spice_preference"] == cust_row["spice_preference"]).mean() > 0.3:
            content_score += 0.1

        collab_score = 0
        orderers = dish_rows["customer_id"].unique()
        sims = [customer_sim_df.loc[customer_id, o] for o in orderers
                if o != customer_id and o in customer_sim_df.columns]
        if sims:
            collab_score = np.mean(sims) * 0.3

        avg_rating = dish_rows["customer_rating"].mean()
        rating_score = (avg_rating / 5) * 0.2

        already_ordered = dish in cust_orders["ordered_item"].values
        recency_score = 0.15 if already_ordered and cust_orders["ordered_again"].mean() > 0.5 else 0

        total = content_score + collab_score + rating_score + recency_score
        dish_scores[dish] = {
            "score": total,
            "content_score": content_score,
            "collab_score": collab_score,
            "rating_score": rating_score,
            "recency_score": recency_score,
            "avg_rating": round(avg_rating, 2)
        }

    ranked = sorted(dish_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]

    results = []
    max_score = max(d[1]["score"] for d in ranked) if ranked else 1

    for dish, s in ranked:
        confidence = round(float(min(s["score"] / max_score * 100, 99.9)), 1)

        signal_strengths = {
            "Frequently ordered by you before": s["recency_score"] / 0.15,
            "Highly rated by customers with similar preferences": s["rating_score"] / 0.2,
            "Customers with similar taste preferred this": s["collab_score"] / 0.3,
            "Matches your preferred cuisine": s["content_score"] / 0.8,
        }
        best_reason = max(signal_strengths, key=signal_strengths.get)
        if signal_strengths[best_reason] == 0:
            best_reason = "Complements your recent orders"

        results.append({
            "food": dish,
            "confidence": confidence,
            "reason": best_reason
        })

    return {"customer_id": int(customer_id), "recommendations": results}


def rebuild_model():
    """Retrain: rebuild customer_profile and similarity matrix from current dataset."""
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


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "Restaurant Recommendation Engine API is running"}


@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: int):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = recommend_dishes(customer_id, top_n=5)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


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