# plots.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_temporal_trend(df: pd.DataFrame, date_col: str = "TransactionMonth"):
    df["period"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period('M')
    agg = df.groupby("period").agg(
        total_premium=("TotalPremium","sum"),
        total_claims=("TotalClaims","sum")
    ).reset_index()
    agg["LossRatio"] = agg["total_claims"] / agg["total_premium"]

    # Premium & claims
    agg.plot(x="period", y=["total_premium","total_claims"], marker='o')
    plt.title("Total Premium & Claims Over Time")
    plt.show()

    # LossRatio
    sns.lineplot(data=agg, x="period", y="LossRatio", marker='o')
    plt.title("LossRatio Over Time (Monthly)")
    plt.xticks(rotation=45)
    plt.show()

def plot_categorical_loss(df: pd.DataFrame, col: str):
    agg = df.groupby(col).agg(
        total_premium=("TotalPremium","sum"),
        total_claims=("TotalClaims","sum")
    ).reset_index()
    agg["LossRatio"] = agg["total_claims"] / agg["total_premium"]
    sns.barplot(data=agg, x=col, y="LossRatio")
    plt.title(f"LossRatio by {col}")
    plt.xticks(rotation=45)
    plt.show()

def plot_box(df: pd.DataFrame, col: str):
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Detection: {col}")
    plt.show()
