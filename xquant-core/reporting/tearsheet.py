import pandas as pd
from ..risk.metrics import get_performance_metrics
from .plots import plot_equity_curve, plot_drawdown
import json


class Tearsheet:
    """
    Generates a full performance tearsheet for a backtest result.
    """

    def __init__(self, results: pd.DataFrame):
        self.results = results
        self.metrics = get_performance_metrics(results)

    def summary(self):
        """
        Print a text summary of the metrics.
        """
        print("\n" + "=" * 40)
        print("       XQUANT PERFORMANCE TEARSHEET")
        print("=" * 40)
        for key, value in self.metrics.items():
            print(f"{key:25}: {value:>12.4f}")
        print("=" * 40)

    def generate_html(self, filename: str = "tearsheet.html"):
        """
        Export a simplified HTML representation.
        """
        html = f"""
        <html>
        <head><title>XQuant Tearsheet</title></head>
        <body>
            <h1>XQuant Performance Tearsheet</h1>
            <table border="1">
                <tr><th>Metric</th><th>Value</th></tr>
                {"".join([f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in self.metrics.items()])}
            </table>
        </body>
        </html>
        """
        with open(filename, "w") as f:
            f.write(html)
        print(f"Tearsheet saved to {filename}")
