"""
DataDoctor AI - Retail Store Analyzer Module
Generalized retail analytics functions for sales, profit, products,
trends, categories, customers, and inventory analysis.
"""

import pandas as pd
import numpy as np


def map_store_columns(df):
    """
    Intelligently maps raw DataFrame columns to standard retail keys
    based on case-insensitive keyword matching.
    """
    column_aliases = {
        "date": [
            "date", "order_date", "sale_date", "invoice_date",
            "timestamp", "transaction_date", "trans_date", "orderdate", "saledate"
        ],
        "product": [
            "product", "product_name", "item", "item_name",
            "description", "sku", "product_id", "productname", "itemname", "product_title"
        ],
        "quantity": [
            "quantity", "qty", "units", "units_sold",
            "quantity_sold", "count", "num_items", "pieces"
        ],
        "price": [
            "price", "unit_price", "selling_price", "item_price",
            "rate", "retail_price", "sales_price"
        ],
        "revenue": [
            "total", "sales", "revenue", "total_amount", "amount",
            "total_sales", "net_sales", "line_total", "total_revenue", "subtotal"
        ],
        "cost": [
            "cost", "unit_cost", "cost_price", "total_cost",
            "cogs", "expense", "purchase_price", "item_cost"
        ],
        "category": [
            "category", "product_category", "department",
            "item_category", "type", "product_type", "group", "class"
        ],
        "customer": [
            "customer", "customer_id", "customer_name", "client",
            "buyer_id", "user_id", "client_id", "customer_code"
        ],
        "stock": [
            "stock", "quantity_in_stock", "inventory", "current_stock",
            "stock_level", "available_stock", "units_in_stock", "stock_qty"
        ],
        "transaction_id": [
            "transaction_id", "order_id", "invoice_no", "invoice_id",
            "receipt_id", "bill_id", "order_number", "trans_id"
        ]
    }

    mapping = {}
    normalized_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    excluded_from_trans = ["employee_id", "empid", "emp_id", "staff_id", "patient_id", "student_id"]
    excluded_from_date = ["joining_date", "leaving_date", "hire_date", "exit_date", "birth_date", "dob"]
    excluded_from_sales = ["salary", "wage", "payroll", "stipend"]

    for key, aliases in column_aliases.items():
        found_col = None
        # 1. Exact alias match
        for alias in aliases:
            if alias in normalized_cols:
                norm_cand = alias
                if key == "transaction_id" and any(exc in norm_cand for exc in excluded_from_trans):
                    continue
                if key == "date" and any(exc in norm_cand for exc in excluded_from_date):
                    continue
                if key in ["revenue", "price", "cost"] and any(exc in norm_cand for exc in excluded_from_sales):
                    continue
                found_col = normalized_cols[alias]
                break

        # 2. Substring fallback match if not yet found
        if not found_col:
            for norm_name, original_name in normalized_cols.items():
                if key == "transaction_id" and any(exc in norm_name for exc in excluded_from_trans):
                    continue
                if key == "date" and any(exc in norm_name for exc in excluded_from_date):
                    continue
                if key in ["revenue", "price", "cost"] and any(exc in norm_name for exc in excluded_from_sales):
                    continue

                for alias in aliases:
                    if len(alias) >= 3 and alias in norm_name and original_name not in mapping.values():
                        found_col = original_name
                        break
                if found_col:
                    break

        if found_col:
            mapping[key] = found_col

    detected = list(mapping.keys())
    missing_critical = [k for k in ["product"] if k not in mapping]
    missing_sales = []
    if "revenue" not in mapping and not ("quantity" in mapping and "price" in mapping):
        missing_sales.append("revenue (or quantity + unit price)")

    return {
        "mapping": mapping,
        "detected": detected,
        "missing_critical": missing_critical + missing_sales,
        "optional_detected": [k for k in ["cost", "category", "customer", "stock", "date"] if k in mapping],
        "optional_missing": [k for k in ["cost", "category", "customer", "stock", "date"] if k not in mapping]
    }


def prepare_sales_data(df, mapping=None):
    """
    Cleans and standardizes sales data using the detected column mapping.
    Computes derived fields (revenue, cost, profit, date components) safely.
    """
    if mapping is None:
        mapping_info = map_store_columns(df)
        mapping = mapping_info["mapping"]

    work_df = df.copy()

    # 1. Quantity
    if "quantity" in mapping:
        work_df["_clean_qty"] = pd.to_numeric(work_df[mapping["quantity"]], errors="coerce").fillna(1)
    else:
        work_df["_clean_qty"] = 1

    # 2. Price
    if "price" in mapping:
        work_df["_clean_price"] = pd.to_numeric(work_df[mapping["price"]], errors="coerce").fillna(0.0)

    # 3. Revenue
    if "revenue" in mapping:
        work_df["_clean_revenue"] = pd.to_numeric(work_df[mapping["revenue"]], errors="coerce")
        # If any NaN, compute from qty * price if price exists
        if "_clean_price" in work_df:
            work_df["_clean_revenue"] = work_df["_clean_revenue"].fillna(work_df["_clean_qty"] * work_df["_clean_price"])
        else:
            work_df["_clean_revenue"] = work_df["_clean_revenue"].fillna(0.0)
    elif "_clean_price" in work_df:
        work_df["_clean_revenue"] = work_df["_clean_qty"] * work_df["_clean_price"]
    else:
        work_df["_clean_revenue"] = 0.0

    # Ensure price is filled if missing but revenue & qty exist
    if "_clean_price" not in work_df:
        work_df["_clean_price"] = np.where(
            work_df["_clean_qty"] > 0,
            work_df["_clean_revenue"] / work_df["_clean_qty"],
            0.0
        )

    # 4. Product Name
    if "product" in mapping:
        work_df["_clean_product"] = work_df[mapping["product"]].astype(str).str.strip()
    else:
        work_df["_clean_product"] = "Unspecified Product"

    # 5. Category
    if "category" in mapping:
        work_df["_clean_category"] = work_df[mapping["category"]].astype(str).str.strip().replace({"nan": "Uncategorized", "None": "Uncategorized", "": "Uncategorized"})
    else:
        work_df["_clean_category"] = "General"

    # 6. Cost and Profit
    has_cost = False
    if "cost" in mapping:
        raw_cost = pd.to_numeric(work_df[mapping["cost"]], errors="coerce").fillna(0.0)
        # Determine if cost column is unit cost or total cost
        col_name_lower = mapping["cost"].lower()
        if "total" in col_name_lower:
            work_df["_clean_cost"] = raw_cost
        else:
            # Assume unit cost
            work_df["_clean_cost"] = raw_cost * work_df["_clean_qty"]

        work_df["_clean_profit"] = work_df["_clean_revenue"] - work_df["_clean_cost"]
        has_cost = True
    else:
        work_df["_clean_cost"] = 0.0
        work_df["_clean_profit"] = 0.0

    # 7. Date components
    has_date = False
    if "date" in mapping:
        work_df["_clean_date"] = pd.to_datetime(work_df[mapping["date"]], errors="coerce")
        valid_dates = work_df["_clean_date"].dropna()
        if len(valid_dates) > 0:
            has_date = True
            work_df["_date_only"] = work_df["_clean_date"].dt.date
            work_df["_year_month"] = work_df["_clean_date"].dt.strftime("%Y-%m")
            work_df["_month_name"] = work_df["_clean_date"].dt.strftime("%b %Y")
            work_df["_day_name"] = work_df["_clean_date"].dt.day_name()
    
    # 8. Customer
    if "customer" in mapping:
        work_df["_clean_customer"] = work_df[mapping["customer"]].astype(str).str.strip()

    # 9. Stock
    if "stock" in mapping:
        work_df["_clean_stock"] = pd.to_numeric(work_df[mapping["stock"]], errors="coerce").fillna(0)

    # 10. Transaction ID
    if "transaction_id" in mapping:
        work_df["_clean_trans_id"] = work_df[mapping["transaction_id"]].astype(str).str.strip()
    else:
        work_df["_clean_trans_id"] = work_df.index.astype(str)

    metadata = {
        "has_date": has_date,
        "has_cost": has_cost,
        "has_category": "category" in mapping,
        "has_customer": "customer" in mapping,
        "has_stock": "stock" in mapping,
        "mapping": mapping
    }

    return work_df, mapping, metadata


def summarize_sales(df, mapping=None):
    """
    Generates high-level retail KPI metrics:
    total revenue, total units, transactions, unique products, profit, margins.
    """
    total_revenue = float(df["_clean_revenue"].sum()) if "_clean_revenue" in df else 0.0
    total_units = int(df["_clean_qty"].sum()) if "_clean_qty" in df else len(df)
    
    if "_clean_trans_id" in df:
        total_transactions = int(df["_clean_trans_id"].nunique())
    else:
        total_transactions = len(df)

    unique_products = int(df["_clean_product"].nunique()) if "_clean_product" in df else 0
    avg_order_value = round(total_revenue / total_transactions, 2) if total_transactions > 0 else 0.0
    avg_unit_price = round(total_revenue / total_units, 2) if total_units > 0 else 0.0

    has_cost = bool("_clean_cost" in df and df["_clean_cost"].sum() > 0)
    total_cost = float(df["_clean_cost"].sum()) if has_cost else 0.0
    total_profit = float(df["_clean_profit"].sum()) if has_cost else 0.0
    profit_margin = round((total_profit / total_revenue) * 100, 2) if (has_cost and total_revenue > 0) else 0.0

    return {
        "total_revenue": round(total_revenue, 2),
        "total_units": total_units,
        "total_transactions": total_transactions,
        "unique_products": unique_products,
        "avg_order_value": avg_order_value,
        "avg_unit_price": avg_unit_price,
        "has_cost": has_cost,
        "total_cost": round(total_cost, 2),
        "total_profit": round(total_profit, 2),
        "profit_margin": profit_margin
    }


def get_top_products(df, mapping=None, top_n=10, sort_by="revenue"):
    """
    Aggregates product performance ranked by revenue or quantity sold.
    """
    if "_clean_product" not in df:
        return pd.DataFrame()

    group_cols = ["_clean_product"]
    if "_clean_category" in df:
        group_cols.append("_clean_category")

    agg_dict = {
        "_clean_qty": "sum",
        "_clean_revenue": "sum",
        "_clean_trans_id": "nunique"
    }

    has_cost = "_clean_profit" in df and df["_clean_cost"].sum() > 0
    if has_cost:
        agg_dict["_clean_profit"] = "sum"

    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()

    grouped = grouped.rename(columns={
        "_clean_product": "Product",
        "_clean_category": "Category",
        "_clean_qty": "Units Sold",
        "_clean_revenue": "Total Revenue",
        "_clean_trans_id": "Orders Count",
        "_clean_profit": "Gross Profit"
    })

    grouped["Total Revenue"] = grouped["Total Revenue"].round(2)
    if has_cost:
        grouped["Gross Profit"] = grouped["Gross Profit"].round(2)
        grouped["Margin %"] = np.where(
            grouped["Total Revenue"] > 0,
            ((grouped["Gross Profit"] / grouped["Total Revenue"]) * 100).round(2),
            0.0
        )

    sort_col = "Total Revenue" if sort_by == "revenue" else "Units Sold"
    top_df = grouped.sort_values(by=sort_col, ascending=False).head(top_n).reset_index(drop=True)
    return top_df


def get_sales_trend(df, mapping=None, freq="daily"):
    """
    Aggregates sales metrics over time by daily or monthly frequency.
    """
    if "_clean_date" not in df or df["_clean_date"].dropna().empty:
        return pd.DataFrame()

    valid_df = df[df["_clean_date"].notna()].copy()
    has_cost = "_clean_profit" in valid_df and valid_df["_clean_cost"].sum() > 0

    if freq == "monthly":
        valid_df["Period"] = valid_df["_clean_date"].dt.to_period("M").dt.to_timestamp()
        label_fmt = "%b %Y"
    else:
        valid_df["Period"] = valid_df["_clean_date"].dt.floor("d")
        label_fmt = "%Y-%m-%d"

    agg_dict = {
        "_clean_revenue": "sum",
        "_clean_qty": "sum",
        "_clean_trans_id": "nunique"
    }
    if has_cost:
        agg_dict["_clean_profit"] = "sum"

    trend = valid_df.groupby("Period").agg(agg_dict).reset_index()
    trend = trend.sort_values("Period")
    trend["Period_Label"] = trend["Period"].dt.strftime(label_fmt)

    trend = trend.rename(columns={
        "_clean_revenue": "Revenue",
        "_clean_qty": "Units Sold",
        "_clean_trans_id": "Transactions",
        "_clean_profit": "Profit"
    })
    trend["Revenue"] = trend["Revenue"].round(2)
    if has_cost:
        trend["Profit"] = trend["Profit"].round(2)

    return trend


def get_category_summary(df, mapping=None):
    """
    Aggregates sales, revenue, and product counts category-wise.
    """
    if "_clean_category" not in df:
        return pd.DataFrame()

    has_cost = "_clean_profit" in df and df["_clean_cost"].sum() > 0

    agg_dict = {
        "_clean_revenue": "sum",
        "_clean_qty": "sum",
        "_clean_product": "nunique",
        "_clean_trans_id": "nunique"
    }
    if has_cost:
        agg_dict["_clean_profit"] = "sum"

    cat_df = df.groupby("_clean_category").agg(agg_dict).reset_index()
    cat_df = cat_df.rename(columns={
        "_clean_category": "Category",
        "_clean_revenue": "Revenue",
        "_clean_qty": "Units Sold",
        "_clean_product": "Unique Products",
        "_clean_trans_id": "Orders",
        "_clean_profit": "Profit"
    })

    total_revenue = cat_df["Revenue"].sum()
    cat_df["Revenue Share %"] = (
        (cat_df["Revenue"] / total_revenue * 100).round(2) if total_revenue > 0 else 0.0
    )
    cat_df["Revenue"] = cat_df["Revenue"].round(2)
    if has_cost:
        cat_df["Profit"] = cat_df["Profit"].round(2)
        cat_df["Margin %"] = np.where(
            cat_df["Revenue"] > 0,
            ((cat_df["Profit"] / cat_df["Revenue"]) * 100).round(2),
            0.0
        )

    cat_df = cat_df.sort_values(by="Revenue", ascending=False).reset_index(drop=True)
    return cat_df


def get_customer_summary(df, mapping=None):
    """
    Analyzes customer purchasing and spending patterns.
    """
    if "_clean_customer" not in df:
        return None

    valid_df = df[df["_clean_customer"].notna() & (df["_clean_customer"] != "")].copy()
    if valid_df.empty:
        return None

    cust_agg = valid_df.groupby("_clean_customer").agg({
        "_clean_revenue": "sum",
        "_clean_qty": "sum",
        "_clean_trans_id": "nunique"
    }).reset_index()

    cust_agg = cust_agg.rename(columns={
        "_clean_customer": "Customer",
        "_clean_revenue": "Total Spend",
        "_clean_qty": "Total Units",
        "_clean_trans_id": "Order Count"
    })

    cust_agg["Total Spend"] = cust_agg["Total Spend"].round(2)
    cust_agg["Avg Spend Per Order"] = (cust_agg["Total Spend"] / cust_agg["Order Count"]).round(2)

    total_customers = len(cust_agg)
    repeat_customers = int((cust_agg["Order Count"] > 1).sum())
    repeat_rate = round((repeat_customers / total_customers) * 100, 2) if total_customers > 0 else 0.0
    avg_customer_value = round(cust_agg["Total Spend"].mean(), 2) if total_customers > 0 else 0.0

    top_customers = cust_agg.sort_values(by="Total Spend", ascending=False).head(10).reset_index(drop=True)

    return {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_rate": repeat_rate,
        "avg_customer_value": avg_customer_value,
        "top_customers": top_customers
    }


def get_inventory_summary(df, mapping=None, low_stock_threshold=10):
    """
    Analyzes inventory levels, low-stock warnings, and stock valuations.
    """
    if "_clean_stock" not in df or "_clean_product" not in df:
        return None

    # Group by product to get latest or minimum stock level
    inv_df = df.groupby("_clean_product").agg({
        "_clean_stock": "last",
        "_clean_price": "mean",
        "_clean_category": "first"
    }).reset_index()

    inv_df = inv_df.rename(columns={
        "_clean_product": "Product",
        "_clean_stock": "Current Stock",
        "_clean_price": "Unit Price",
        "_clean_category": "Category"
    })

    inv_df["Current Stock"] = inv_df["Current Stock"].astype(int)
    inv_df["Unit Price"] = inv_df["Unit Price"].round(2)
    inv_df["Estimated Stock Value"] = (inv_df["Current Stock"] * inv_df["Unit Price"]).round(2)

    total_stock_units = int(inv_df["Current Stock"].sum())
    total_stock_value = round(float(inv_df["Estimated Stock Value"].sum()), 2)
    out_of_stock = inv_df[inv_df["Current Stock"] <= 0]
    low_stock = inv_df[(inv_df["Current Stock"] > 0) & (inv_df["Current Stock"] <= low_stock_threshold)]
    healthy_stock = inv_df[inv_df["Current Stock"] > low_stock_threshold]

    # Assign status tag
    def assign_status(qty):
        if qty <= 0:
            return "Out of Stock"
        elif qty <= low_stock_threshold:
            return "Low Stock"
        return "In Stock"

    inv_df["Status"] = inv_df["Current Stock"].apply(assign_status)

    return {
        "total_stock_units": total_stock_units,
        "total_stock_value": total_stock_value,
        "out_of_stock_count": len(out_of_stock),
        "low_stock_count": len(low_stock),
        "healthy_stock_count": len(healthy_stock),
        "low_stock_products": low_stock.sort_values(by="Current Stock").reset_index(drop=True),
        "out_of_stock_products": out_of_stock.reset_index(drop=True),
        "inventory_table": inv_df.sort_values(by="Current Stock").reset_index(drop=True)
    }
