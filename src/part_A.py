import os
import sqlite3
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(PROJECT_ROOT, "data")
clean_dir = os.path.join(data_dir, "clean")

os.makedirs(clean_dir, exist_ok=True)
db_path = os.path.join(data_dir, "instamart.db")
    
    # 1. Audit & Load
df_cust = pd.read_csv(os.path.join(data_dir, "raw", "customers.csv"))
df_prod = pd.read_csv(os.path.join(data_dir, "raw", "products.csv"))
df_orders = pd.read_csv(os.path.join(data_dir, "raw", "orders.csv"))

print("\nCustomers:")
print(df_cust.head())
print(df_cust.shape)
print("\nNull Values:")
print(df_cust.isnull().sum())

print("\nProducts:")
print(df_prod.head())
print( df_prod.shape)
print("\nNull Values:")
print(df_prod.isnull().sum())

print("\nOrders:")
print(df_orders.head())
print(df_orders.shape)
print("\nNull Values:")
print(df_orders.isnull().sum())



    
# 2. Clean Customers

# Standardize city
df_cust["city"] = df_cust["city"].astype(str).str.strip().str.title()

# datetime format
df_cust["signup_date"] = pd.to_datetime(df_cust["signup_date"])
       
# Clean age: treat outside 10-100 as NaN, then impute with city median
df_cust["age"] = pd.to_numeric(df_cust["age"], errors="coerce")
df_cust.loc[(df_cust["age"] < 10) | (df_cust["age"] > 100), "age"] = np.nan
df_cust["age"] = df_cust.groupby("city")["age"].transform(lambda x: x.fillna(x.median()))

# Standardize gender
gender_map = {
        "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
        "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female","": "unknown"
    }


    
# 3. Clean Orders

# datetime format
df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])

# Drop exact duplicates
df_orders = df_orders.drop_duplicates()

# Sort and keep latest order_ts per order_id
df_orders = df_orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    
# Clean order_amount: coerce numeric, then backfill missing via formula
df_orders["order_amount"] = pd.to_numeric(df_orders["order_amount"], errors="coerce")
df_merged = df_orders.merge(df_prod[["product_id", "price_inr"]], on="product_id", how="left")
calc_amount = (df_merged["price_inr"] * df_merged["quantity"] * (1 - df_merged["discount_pct"] / 100.0)).round(2)
df_orders["order_amount"] = df_orders["order_amount"].fillna(calc_amount)
    
# Clean delivery_minutes
df_orders["delivery_minutes"] = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
df_orders.loc[df_orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    
# Merge category for median calculation
df_orders = df_orders.merge(df_prod[["product_id", "category"]], on="product_id", how="left")
completed_mask = df_orders["status"] == "completed"
    
median_deliv = df_orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
df_orders.loc[completed_mask, "delivery_minutes"] = df_orders.loc[completed_mask, "delivery_minutes"].fillna(median_deliv)

# Ensure cancelled orders have null delivery_minutes
df_orders.loc[df_orders["status"] == "cancelled", "delivery_minutes"] = np.nan
df_orders = df_orders.drop(columns=["category"])
    
# 4. Save Clean Datasets & SQLite Export
df_cust.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)
df_prod.to_csv(os.path.join(clean_dir, "products.csv"), index=False)
df_orders.to_csv(os.path.join(clean_dir, "orders.csv"), index=False)
    
# Format order_ts back to string for SQLite consistency
df_orders_db = df_orders.copy()
df_orders_db["order_ts"] = df_orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
df_cust_db = df_cust.copy()
df_cust_db["signup_date"] = df_cust_db["signup_date"].dt.strftime("%Y-%m-%d")
    
conn = sqlite3.connect(db_path)
df_cust_db.to_sql("customers", conn, if_exists="replace", index=False)
df_prod.to_sql("products", conn, if_exists="replace", index=False)
df_orders_db.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
    
# 5. Validation Checkpoints
print("Total Rows:", len(df_orders))
print("Exact Duplicates:",df_orders.duplicated().sum())
print("Pending Orders:",(df_orders["status"] == "pending").sum())
print("Distinct Order IDs:",df_orders["order_id"].nunique())
invalid_amounts = pd.to_numeric(df_orders["order_amount"],errors="coerce").isna().sum()
print("Invalid Amounts:", invalid_amounts)

