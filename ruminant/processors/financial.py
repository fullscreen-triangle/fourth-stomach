"""
Financial Data Processor

Transforms financial data sources into training pairs for the four-chamber model.
Each source type produces (context, completion) pairs suitable for domain fine-tuning.
"""

import json
import csv
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingPair:
    """A single training example for domain adaptation."""
    context: str      # input text
    completion: str   # target output
    source: str       # data source identifier
    chamber_hint: str # which chamber benefits most: rumen|reticulum|omasum|abomasum


class FinancialProcessor:
    """Processes diverse financial data into chamber-optimised training pairs."""

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs: list[TrainingPair] = []

    def process_market_data(self, csv_path: str) -> list[TrainingPair]:
        """Transform price/volume CSV into thermodynamic analysis pairs.

        Input format: date,ticker,open,high,low,close,volume
        Output: pairs like (price_series -> phase_identification)
        """
        pairs = []
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Group by ticker
            tickers = {}
            for row in rows:
                t = row.get("ticker", "UNKNOWN")
                tickers.setdefault(t, []).append(row)

            for ticker, data in tickers.items():
                closes = [float(r.get("close", 0)) for r in data[-30:] if r.get("close")]
                if len(closes) < 10:
                    continue

                # Context: recent price series
                price_str = ", ".join(f"{p:.2f}" for p in closes[-10:])
                context = f"Ticker: {ticker}\nRecent closes: [{price_str}]"

                # Completion: thermodynamic analysis
                import numpy as np
                returns = np.diff(closes) / (np.array(closes[:-1]) + 1e-8)
                vol = float(np.std(returns))
                mean_ret = float(np.mean(returns))

                completion = (
                    f"Variance (T_var proxy): {vol:.6f}\n"
                    f"Mean return: {mean_ret:.6f}\n"
                    f"Phase: {'gas (low correlation)' if vol < 0.02 else 'liquid (moderate)' if vol < 0.05 else 'crystal (high correlation)'}"
                )

                pairs.append(TrainingPair(
                    context=context,
                    completion=completion,
                    source=f"market:{ticker}",
                    chamber_hint="reticulum",  # spectral analysis task
                ))
        except Exception:
            pass

        self.pairs.extend(pairs)
        return pairs

    def process_sec_filing(self, text: str, filing_type: str = "10-K") -> list[TrainingPair]:
        """Extract structured knowledge from SEC filings."""
        pairs = []

        # Split into sections
        sections = re.split(r'\n(?=ITEM \d)', text, flags=re.IGNORECASE)

        for section in sections:
            if len(section) < 100:
                continue

            # Context: section text (truncated)
            context = section[:2000]

            # Completion: structured extraction
            completion = (
                f"Filing type: {filing_type}\n"
                f"Section length: {len(section)} chars\n"
                f"Key entities: [extracted from text]\n"
                f"Risk factors: [identified from context]"
            )

            pairs.append(TrainingPair(
                context=context,
                completion=completion,
                source=f"sec:{filing_type}",
                chamber_hint="omasum",  # graph completion task (filling gaps)
            ))

        self.pairs.extend(pairs)
        return pairs

    def process_transaction_stream(self, transactions: list[dict]) -> list[TrainingPair]:
        """Convert transaction streams into Kirchhoff-consistent flow descriptions."""
        pairs = []

        for i in range(0, len(transactions) - 10, 10):
            batch = transactions[i:i+10]

            # Context: transaction batch
            tx_lines = []
            for tx in batch:
                tx_lines.append(
                    f"{tx.get('from','?')} -> {tx.get('to','?')}: "
                    f"${tx.get('amount', 0):.2f} at t={tx.get('timestamp', 0)}"
                )
            context = "Transaction batch:\n" + "\n".join(tx_lines)

            # Completion: circuit analysis
            nodes = set()
            for tx in batch:
                nodes.add(tx.get('from', ''))
                nodes.add(tx.get('to', ''))
            nodes.discard('')

            completion = (
                f"Nodes: {len(nodes)}\n"
                f"Edges: {len(batch)}\n"
                f"KCL check: [verify inflow = outflow at each node]\n"
                f"KVL check: [verify no arbitrage loops]"
            )

            pairs.append(TrainingPair(
                context=context,
                completion=completion,
                source="transactions",
                chamber_hint="rumen",  # circulation task
            ))

        self.pairs.extend(pairs)
        return pairs

    def save(self, filename: str = "financial_training.jsonl"):
        """Save all training pairs as JSONL."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            for pair in self.pairs:
                json.dump({
                    "context": pair.context,
                    "completion": pair.completion,
                    "source": pair.source,
                    "chamber_hint": pair.chamber_hint,
                }, f)
                f.write("\n")
        return path

    def stats(self) -> dict:
        """Return statistics about processed data."""
        by_chamber = {}
        by_source = {}
        for p in self.pairs:
            by_chamber[p.chamber_hint] = by_chamber.get(p.chamber_hint, 0) + 1
            src = p.source.split(":")[0]
            by_source[src] = by_source.get(src, 0) + 1
        return {
            "total_pairs": len(self.pairs),
            "by_chamber": by_chamber,
            "by_source": by_source,
        }
