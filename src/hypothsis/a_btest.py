import pandas as pd
from scipy import stats
from src.utility.config_loader import loader as config_loader
from src.utility.data_loader import loader as data_loader
class ABTest:
    """
    Class to run hypothesis tests for insurance data.
    Hypotheses:
        1. Risk difference across Provinces
        2. Risk difference across PostalCodes
        3. Margin difference across PostalCodes
        4. Risk difference by Gender
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Policy-level dataframe containing at least these columns:
            - PolicyID
            - TotalClaims
            - TotalPremium
            - Province
            - PostalCode
            - Gender
        """
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Aggregate to policy level and create derived columns."""
        # If not already aggregated by PolicyID, aggregate
        if "PolicyID" in self.df.columns:
            agg = self.df.groupby("PolicyID").agg(
                TotalClaims=("TotalClaims", "sum"),
                TotalPremium=("TotalPremium", "sum"),
                Province=("Province", "first"),
                PostalCode=("PostalCode", "first"),
                Gender=("Gender", "first")
            ).reset_index()
            self.policy_df = agg
        else:
            self.policy_df = self.df.copy()

        # Derived columns
        self.policy_df["has_claim"] = (self.policy_df["TotalClaims"] > 0).astype(int)
        self.policy_df["margin"] = self.policy_df["TotalPremium"] - self.policy_df["TotalClaims"]

    def run_tests(self) -> pd.DataFrame:
        """Run 4 hypothesis tests and return a summary table."""
        results = []

        # H1: Risk by Province
        table = pd.crosstab(self.policy_df["Province"], self.policy_df["has_claim"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        results.append({
            "Hypothesis": "Risk by Province",
            "Test": "Chi-square",
            "p_value": p,
            "Result": "Reject" if p < 0.05 else "Fail to reject",
            "Interpretation": "Some provinces have higher claim frequency; adjust premiums regionally" if p < 0.05 else "No significant difference across provinces"
        })

        # H2: Risk by PostalCode (use top 20 codes for stability)
        top_postal = self.policy_df["PostalCode"].value_counts().nlargest(20).index
        df_pc = self.policy_df[self.policy_df["PostalCode"].isin(top_postal)]
        table = pd.crosstab(df_pc["PostalCode"], df_pc["has_claim"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        results.append({
            "Hypothesis": "Risk by PostalCode",
            "Test": "Chi-square",
            "p_value": p,
            "Result": "Reject" if p < 0.05 else "Fail to reject",
            "Interpretation": "Claim frequency differs by postal code; consider zip-based segmentation" if p < 0.05 else "No significant difference across postal codes"
        })

        # H3: Margin by top 2 PostalCodes
        top2_postal = self.policy_df["PostalCode"].value_counts().nlargest(2).index
        group1 = self.policy_df[self.policy_df["PostalCode"]==top2_postal[0]]["margin"]
        group2 = self.policy_df[self.policy_df["PostalCode"]==top2_postal[1]]["margin"]
        t_stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        results.append({
            "Hypothesis": "Margin by PostalCode",
            "Test": "T-test",
            "p_value": p,
            "Result": "Reject" if p < 0.05 else "Fail to reject",
            "Interpretation": "Significant margin difference; adjust premiums" if p < 0.05 else "No significant difference in margin"
        })

        # H4: Risk by Gender
        table = pd.crosstab(self.policy_df["Gender"], self.policy_df["has_claim"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        results.append({
            "Hypothesis": "Risk by Gender",
            "Test": "Chi-square",
            "p_value": p,
            "Result": "Reject" if p < 0.05 else "Fail to reject",
            "Interpretation": "Claim frequency differs by gender" if p < 0.05 else "No significant difference between genders"
        })

        return pd.DataFrame(results)

# Example usage (for your notebook or script)
if __name__ == "__main__":
    data_path = config_loader.load('data.yaml')['data']['raw_data_path']

    df = data_loader.load_csv(data_path)
    ab = ABTest(df)
    summary = ab.run_tests()
    print(summary)
