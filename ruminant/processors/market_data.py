"""
Market Data Processor

Transforms raw market data (JSON/CSV validation results, panel figures, etc.)
into training pairs that teach the model numerical reasoning about the framework.
"""

import json
from pathlib import Path
from ruminant.processors.financial import TrainingPair


class MarketDataProcessor:
    """Processes experimental validation results into training pairs."""

    def __init__(self):
        self.pairs: list[TrainingPair] = []

    def process_validation_results(self, results_dir: str) -> list[TrainingPair]:
        """Convert JSON experiment results into reasoning pairs."""
        results_path = Path(results_dir)
        pairs = []

        for json_file in results_path.glob("*.json"):
            if json_file.name == "summary.json":
                continue

            try:
                data = json.loads(json_file.read_text())
            except Exception:
                continue

            experiment = data.get("experiment", json_file.stem)
            prediction = data.get("prediction", "")
            passed = data.get("passed", False)

            # Create a reasoning pair about the experiment
            context = f"Experiment: {experiment}\nPrediction: {prediction}\nDid it pass validation?"
            completion = f"Result: {'PASSED' if passed else 'FAILED'}."

            # Add numerical details if available
            for key in ["mean_ratio", "correlation_T_vs_ratio", "win_rate",
                        "mean_contraction_rate", "max_drift_overall",
                        "parameter_savings", "critical_ratio"]:
                if key in data:
                    completion += f"\n{key}: {data[key]}"

            pairs.append(TrainingPair(
                context=context,
                completion=completion,
                source=f"validation:{json_file.stem}",
                chamber_hint="abomasum",  # numerical reasoning
            ))

            # Create pairs from individual data points
            if "data" in data and isinstance(data["data"], list):
                for i, row in enumerate(data["data"][:5]):  # limit per experiment
                    if isinstance(row, dict):
                        row_str = ", ".join(f"{k}={v}" for k, v in row.items()
                                          if not isinstance(v, (list, dict)))
                        pairs.append(TrainingPair(
                            context=f"Given {experiment} data point: {row_str}\nInterpret this result.",
                            completion=f"This data point from the {experiment} experiment shows {prediction}.",
                            source=f"datapoint:{json_file.stem}:{i}",
                            chamber_hint="reticulum",
                        ))

        self.pairs.extend(pairs)
        return pairs
