import pandas as pd
from typing import Optional


class NumeraiExporter:
    """
    Exports signals in NumerAI submission format.
    """

    @staticmethod
    def export(
        df: pd.DataFrame,
        prediction_col: str,
        id_col: str = "id",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Format DataFrame for NumerAI.

        Args:
            df: Input DataFrame.
            prediction_col: Column containing the signal/prediction [0, 1].
            id_col: Column containing NumerAI IDs.
            output_path: Optional path to save CSV.
        """
        submission = df[[id_col, prediction_col]].copy()
        submission.columns = ["id", "prediction"]

        if output_path:
            submission.to_csv(output_path, index=False)

        return submission
