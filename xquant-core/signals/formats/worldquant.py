import pandas as pd
from typing import Optional


class WorldQuantExporter:
    """
    Exports signals in WorldQuant Brain submission format.
    """

    @staticmethod
    def export(
        df: pd.DataFrame,
        alpha_col: str,
        asset_col: str = "asset",
        date_col: str = "date",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Format DataFrame for WorldQuant Brain.

        Args:
            df: Input DataFrame (tidy format).
            alpha_col: Column containing alpha values.
            asset_col: Column containing asset identifiers.
            date_col: Column containing timestamps.
            output_path: Optional path to save CSV.
        """
        submission = df[[date_col, asset_col, alpha_col]].copy()
        submission.columns = ["date", "asset", "alpha"]

        # WorldQuant Brain often expects YYYY-MM-DD
        submission["date"] = pd.to_datetime(submission["date"]).dt.strftime("%Y-%m-%d")

        if output_path:
            submission.to_csv(output_path, index=False)

        return submission
