# Modular tool definitions for the MCP server
# Each tool is defined here with its metadata

TOOLS_METADATA = [
    # ===== SPENDING TOOLS (ML-powered) =====
    {
        "name": "get_spending_summary",
        "description": "Get total spending and top merchant categories for a client over the last N months with ML anomaly detection.",
        "parameters": {"client_id": "int", "months": "int"},
        "category": "spending"
    },
    {
        "name": "spending_by_category",
        "description": "ML-based spending breakdown by merchant category with clustering insights.",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    {
        "name": "monthly_spending_trend",
        "description": "ML-powered monthly spending trend analysis using linear regression.",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    {
        "name": "spending_by_city",
        "description": "Spending breakdown by merchant city/location with risk assessment.",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    {
        "name": "spending_by_card_type",
        "description": "Spending breakdown by card type (Credit/Debit/Prepaid).",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    {
        "name": "total_transactions_count",
        "description": "Get total number of transactions for a client.",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    {
        "name": "average_transaction_amount",
        "description": "Calculate average transaction amount with ML outlier detection.",
        "parameters": {"client_id": "int"},
        "category": "spending"
    },
    
    # ===== FRAUD TOOLS (ML-powered with Random Forest) =====
    {
        "name": "check_fraud",
        "description": "Estimate fraud probability for a transaction using ML Random Forest classifier.",
        "parameters": {"transaction_id": "int"},
        "category": "fraud"
    },
    {
        "name": "client_fraud_risk",
        "description": "Calculate overall fraud risk score for a client using ML model.",
        "parameters": {"client_id": "int"},
        "category": "fraud"
    },
    {
        "name": "fraud_rate_by_merchant",
        "description": "Get fraud rate by merchant category (MCC).",
        "parameters": {"mcc_code": "str"},
        "category": "fraud"
    },
    {
        "name": "high_risk_transactions",
        "description": "List all high-value transactions with ML risk assessment.",
        "parameters": {"client_id": "int", "threshold": "float"},
        "category": "fraud"
    },
    
    # ===== CREDIT TOOLS =====
    {
        "name": "credit_score_history",
        "description": "Get credit score with grade (excellent/good/fair/poor).",
        "parameters": {"client_id": "int"},
        "category": "credit"
    },
    {
        "name": "debt_income_analysis",
        "description": "Debt to income ratio analysis with risk level.",
        "parameters": {"client_id": "int"},
        "category": "credit"
    },
    {
        "name": "credit_limit_analysis",
        "description": "Credit limit and utilization analysis.",
        "parameters": {"client_id": "int"},
        "category": "credit"
    },
    
    # ===== FORECAST TOOLS (ML-powered) =====
    {
        "name": "spending_forecast",
        "description": "Forecast spending for next N months using ML time-series analysis (Linear Regression + Exponential Smoothing).",
        "parameters": {"client_id": "int", "months": "int"},
        "category": "forecast"
    },
    {
        "name": "monthly_spending_trend",
        "description": "ML-based monthly spending trend with R-squared analysis.",
        "parameters": {"client_id": "int"},
        "category": "forecast"
    },
    {
        "name": "spending_forecast_by_category",
        "description": "Forecast spending by category using ML time series.",
        "parameters": {"client_id": "int", "months": "int"},
        "category": "forecast"
    },
    
    # ===== VISUALIZATION TOOLS (ML-powered) =====
    {
        "name": "transaction_heatmap",
        "description": "ML-powered heatmap of spending by day of week and hour with peak detection.",
        "parameters": {"client_id": "int"},
        "category": "visualization"
    },
    {
        "name": "peak_hours_analysis",
        "description": "Analyze peak transaction hours using ML clustering.",
        "parameters": {"client_id": "int"},
        "category": "visualization"
    },
    {
        "name": "spending_distribution",
        "description": "Distribution of transaction amounts with ML outlier detection.",
        "parameters": {"client_id": "int"},
        "category": "visualization"
    },
    {
        "name": "customer_segmentation",
        "description": "Segment customer using ML clustering analysis.",
        "parameters": {"client_id": "int"},
        "category": "visualization"
    },
    
    # ===== MERCHANT TOOLS (ML-powered) =====
    {
        "name": "merchant_summary",
        "description": "Get total transactions, revenue, and fraud rate for a merchant with ML risk scoring.",
        "parameters": {"mcc_code": "str"},
        "category": "merchant"
    },
    {
        "name": "merchant_risk_analysis",
        "description": "Analyze risk level of a merchant category using ML Random Forest.",
        "parameters": {"mcc_code": "str"},
        "category": "merchant"
    },
    {
        "name": "top_merchants",
        "description": "Get top merchants by total spending for a client with ML insights.",
        "parameters": {"client_id": "int", "limit": "int"},
        "category": "merchant"
    },
    
    # ===== CARD TOOLS =====
    {
        "name": "card_usage_analysis",
        "description": "Analyze card usage patterns (chip vs swipe, card type).",
        "parameters": {"client_id": "int"},
        "category": "card"
    },
    {
        "name": "card_type_spending",
        "description": "Spending breakdown by card brand (Visa/Mastercard/Discover).",
        "parameters": {"client_id": "int"},
        "category": "card"
    },
    
    # ===== PROFILE TOOLS =====
    {
        "name": "customer_profile",
        "description": "Get complete customer profile (age, income, debt, credit).",
        "parameters": {"client_id": "int"},
        "category": "profile"
    },
    {
        "name": "income_analysis",
        "description": "Analyze income and spending ratio.",
        "parameters": {"client_id": "int"},
        "category": "profile"
    },
    
    # ===== ALL-USERS AGGREGATE TOOLS =====
    {
        "name": "get_all_users_spending_summary",
        "description": "Get spending summary for ALL users combined - aggregate analytics.",
        "parameters": {"months": "int"},
        "category": "aggregate"
    },
    {
        "name": "spending_by_category_all",
        "description": "Spending breakdown by merchant category for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "monthly_spending_trend_all",
        "description": "Monthly spending trend for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "spending_by_city_all",
        "description": "Spending breakdown by city for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "spending_by_card_type_all",
        "description": "Spending by card type for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "total_transactions_all",
        "description": "Total transactions count for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "average_transaction_all",
        "description": "Average transaction amount for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "global_fraud_analysis",
        "description": "Global fraud analysis for ALL transactions.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "all_clients_fraud_summary",
        "description": "Fraud summary for ALL clients.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "credit_score_distribution_all",
        "description": "Credit score distribution for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "debt_income_all",
        "description": "Debt to income analysis for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "credit_limit_all",
        "description": "Credit limit and utilization for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "transaction_heatmap_all",
        "description": "Transaction heatmap for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "peak_hours_all",
        "description": "Peak transaction hours for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "spending_distribution_all",
        "description": "Spending distribution for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "merchant_summary_all",
        "description": "Merchant summary for ALL merchants.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "top_merchants_all",
        "description": "Top merchants by spending for ALL users.",
        "parameters": {"limit": "int"},
        "category": "aggregate"
    },
    {
        "name": "card_usage_all",
        "description": "Card usage patterns for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "customer_segmentation_all",
        "description": "Customer segmentation for ALL users.",
        "parameters": {},
        "category": "aggregate"
    },
    {
        "name": "spending_forecast_all",
        "description": "Spending forecast for ALL users.",
        "parameters": {"months": "int"},
        "category": "aggregate"
    },
]

# Group tools by category
TOOLS_BY_CATEGORY = {}
for tool in TOOLS_METADATA:
    cat = tool.get("category", "other")
    if cat not in TOOLS_BY_CATEGORY:
        TOOLS_BY_CATEGORY[cat] = []
    TOOLS_BY_CATEGORY[cat].append(tool["name"])

def get_tools_by_category(category: str) -> list:
    """Get all tool names for a specific category."""
    return TOOLS_BY_CATEGORY.get(category, [])

def get_all_tool_names() -> list:
    """Get list of all tool names."""
    return [t["name"] for t in TOOLS_METADATA]
