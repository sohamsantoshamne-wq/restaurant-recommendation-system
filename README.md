# 🍽️ Restaurant Food Recommendation Engine

An intelligent, explainable AI microservice that generates personalized dish recommendations for restaurant customers based on their order history, preferences, and behavioral patterns.

Built as part of the AD Tech Enterprises Internship Assignment (Task 03).

---

## 📌 Project Overview

When a customer revisits the restaurant, this system identifies them and generates **personalized, explainable food recommendations** instead of showing a generic menu to everyone. It's built as an independent AI microservice, ready to integrate with a Restaurant Management System.

---

## 🗂️ Project Structure

---

## 🧠 Machine Learning Approach

**Hybrid Recommendation System** (Content-Based + Collaborative Filtering)

| Component | What it does |
|---|---|
| **Content-Based** | Matches dishes to a customer's own preferences — cuisine, food category, veg/non-veg, spice level |
| **Collaborative Filtering** | Uses cosine similarity between customer profiles to recommend dishes liked by *similar* customers |
| **Rating Boost** | Favors dishes with consistently high customer ratings |
| **Recency Signal** | Slightly favors dishes the customer has ordered before and rated well |

### Why Hybrid?
- **Content-based alone** struggles with new dishes/customers with sparse data and can get repetitive.
- **Collaborative filtering alone** suffers from the cold-start problem for new customers.
- **Combining both** balances personalization with variety, and remains robust even for customers with limited order history — matching the assignment's explainability and quality requirements.

Each recommendation includes a **confidence score (0–100)** and a **human-readable reason**, generated from whichever signal (content match, similar-customer preference, high rating, or reorder pattern) contributed most to that dish's score.

---

## 📊 Synthetic Dataset

- **8,000 customer interaction records** across **1,500 unique customers**
- Generated using `pandas`, `numpy`, and `faker` with **realistic behavioral patterns** (not pure randomness) — e.g., dinner-time orders skew toward mains, customers matching their preferred cuisine reorder more often, frequent visitors have higher reorder probability.
- Fields: customer ID, age group, gender, visit frequency, preferred cuisine, food category, previous orders, frequently/recently ordered items, average bill, time of visit, day of week, season, veg/non-veg & spice preference, rating, timestamp, and `ordered_again` (target variable).

---

## 📈 Model Evaluation

Evaluated on a sample of 190 customers using Precision@5, Recall@5, and F1 Score, where "ground truth" is defined as dishes a customer rated 4★+ or explicitly reordered.

| Metric | Score |
|---|---|
| **Precision@5** | 0.4874 |
| **Recall@5** | 0.8369 |
| **F1 Score** | 0.6160 |
| **Accuracy** | 0.7482 |
| Evaluated on | 190 customers |

*Note: RMSE was not used, as this is a ranking/classification-style recommendation task rather than a continuous value prediction. RMSE is not applicable here.*

**Interpretation:** Roughly half of the top-5 recommendations directly match dishes the customer is known to like, and the model successfully surfaces ~84% of all dishes a customer would enjoy within its top-5 list — a strong result for a synthetic-data MVP.

---

## 🔌 REST API

Built with **FastAPI**.

### `GET /recommend/{customer_id}`
Returns the top 5 personalized dish recommendations.

**Example Response:**
```json
{
  "customer_id": 1023,
  "recommendations": [
    {
      "food": "Paneer Butter Masala",
      "confidence": 96.4,
      "reason": "Frequently ordered during dinner."
    },
    {
      "food": "Garlic Naan",
      "confidence": 93.1,
      "reason": "Commonly purchased with paneer dishes."
    }
  ]
}
```

### `POST /order`
Stores a new customer order for future model retraining.

### `POST /train`
Retrains the recommendation engine using the latest dataset.

Interactive API docs available at: `http://127.0.0.1:8000/docs`

---

## 📊 Analytics Dashboard

A Streamlit dashboard (`dashboard/dashboard.py`) visualizing:
- Total customers & orders
- Most ordered items
- Cuisine popularity
- Customer visit distribution
- Category distribution
- Revenue distribution
- Time-of-visit breakdown
- Model recommendation accuracy

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/restaurant-recommendation-system.git
cd restaurant-recommendation-system
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset & train the model
Run the notebooks in order: