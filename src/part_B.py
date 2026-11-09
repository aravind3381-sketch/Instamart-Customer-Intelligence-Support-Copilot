# src/part_b_analytics.py
import os
import profile
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")



os.makedirs(OUTPUT_DIR, exist_ok=True)
clean_dir = os.path.join(DATA_DIR, "clean")
df_cust = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
df_orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))
df_prod = pd.read_csv(os.path.join(clean_dir, "products.csv"))
    
df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])
df_cust["signup_date"] = pd.to_datetime(df_cust["signup_date"])    
    # --- Task 1: RFM Table ---
snapshot = pd.Timestamp("2025-10-01 00:00:00")
completed_order = df_orders[df_orders["status"] == "completed"]
    
rfm = completed_order.groupby("customer_id").agg(
        recency=("order_ts", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
print("RFM Shape:", rfm.shape)

print("\nRecency")
print(rfm["recency"].describe())

print("\nFrequency")
print(rfm["frequency"].describe())

print("\nMonetary")
print(rfm["monetary"].describe())
    
    # --- Task 2: K-Means Clustering ---
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
# kmeans clustering for k=2 to k=8
for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(
        rfm_scaled
    )

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        rfm_scaled,
        labels
    )

    print(
        f"K={k} | "
        f"Inertia={inertia:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

    # Fit K=4
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(rfm_scaled)
    
    # Profile & Mark Premium (highest mean monetary)
cluster_profiles = rfm.groupby("segment_id").agg(
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        count=("customer_id", "count")
    ).round(2)

print("\n===== SEGMENT PROFILE =====")
print(cluster_profiles)
    

# Map cluster to segment names
segment_names = {
    0: "Occasional Shoppers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At Risk Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
premium_cluster = cluster_profiles["mean_monetary"].idxmax()

rfm["is_premium"] = (
    rfm["segment_id"] == premium_cluster
)
    # PCA check
pca = PCA(n_components=2)
pca.fit(rfm_scaled)
    
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
        os.path.join(OUTPUT_DIR, "customer_segments.csv"), index=False
    )
    
    # --- Task 4 & 5: Lapse Modeling Dataset ---
cutoff = pd.Timestamp("2025-09-01 00:00:00")
prior_orders = df_orders[df_orders["order_ts"] < cutoff]
post_orders = df_orders[(df_orders["order_ts"] >= cutoff) & (df_orders["order_ts"] <= pd.Timestamp("2025-09-30 23:59:59"))]
    
    # Eligible cohort: completed order before cutoff
prior_completed = prior_orders[prior_orders["status"] == "completed"]
cohort_ids = prior_completed["customer_id"].unique()
    
    # Label: 1 if no completed order in Sept 2025
post_completed_ids = set(post_orders[post_orders["status"] == "completed"]["customer_id"].unique())
lapsed_map = {cid: (1 if cid not in post_completed_ids else 0) for cid in cohort_ids}
    
    # Prior order features
prior_orders_with_cat = prior_orders.merge(df_prod[["product_id", "category"]], on="product_id", how="left")
    
features = []
for cid in cohort_ids:
    c_orders = prior_orders_with_cat[prior_orders_with_cat["customer_id"] == cid]
    c_comp = c_orders[c_orders["status"] == "completed"]
        
    rec = (cutoff - c_comp["order_ts"].max()).days
    freq = len(c_comp)
    mon = c_comp["order_amount"].sum()
    avg_disc = c_comp["discount_pct"].mean()
    canc_rate = (c_orders["status"] == "cancelled").sum() / len(c_orders)
    avg_deliv = c_comp["delivery_minutes"].mean()
    fav_cat = c_comp["category"].mode()[0] if not c_comp.empty else "Grocery"
        
    features.append({
        "customer_id": cid,
        "recency": rec,
        "frequency": freq,
        "monetary": mon,
        "avg_discount": avg_disc,
        "cancellation_rate": canc_rate,
        "avg_delivery_minutes": avg_deliv,
        "favourite_category": fav_cat,
        "lapsed": lapsed_map[cid]
        })
        
df_lapse = pd.DataFrame(features).merge(df_cust, on="customer_id", how="left")
df_lapse["tenure_days"] = (cutoff - df_lapse["signup_date"]).dt.days
    
num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
cat_cols = ["city", "gender", "favourite_category"]
    
X = df_lapse[num_cols + cat_cols]
y = df_lapse["lapsed"]
    
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, df_lapse["customer_id"], test_size=0.2, stratify=y, random_state=42
    )
    
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )
    
    # --- Task 6: Tune Models ---
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
rf_params = {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)
    
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
lr_params = {"clf__C": [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr_pipe, lr_params, cv=5, scoring="f1", n_jobs=-1)
lr_grid.fit(X_train, y_train)
    
    # Select best model (Random Forest)

best_model = rf_grid.best_estimator_
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "lapse_model.joblib"))
    
    # --- Task 8: Retention List ---
test_probs = best_model.predict_proba(X_test)[:, 1]
df_test_res = pd.DataFrame({
    "customer_id": id_test,
    "lapse_probability": test_probs,
    "actually_lapsed": y_test.values
    }).merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
    
premium_test = df_test_res[df_test_res["is_premium"] == True].sort_values("lapse_probability", ascending=False)
retention_list = premium_test.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
retention_list.to_csv(os.path.join(OUTPUT_DIR, "retention_list.csv"), index=False)
    
print("PART B COMPLETE: Model & segments persisted.")
