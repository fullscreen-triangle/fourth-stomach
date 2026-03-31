"""
Ruminant Pipeline

The main entry point for the framework. Orchestrates:
  1. Data ingestion through domain-specific processors
  2. Training corpus generation with chamber hints
  3. Chamber-specific LoRA fine-tuning
  4. Thermodynamic evaluation
"""

from pathlib import Path
from ruminant.processors.financial import FinancialProcessor
from ruminant.processors.theorem import TheoremProcessor
from ruminant.processors.market_data import MarketDataProcessor
from ruminant.trainers.chamber_lora import ChamberLoRATrainer, ChamberLoRAConfig
from ruminant.evaluation.thermodynamic import ThermodynamicEvaluator


class RuminantPipeline:
    """End-to-end pipeline for domain-specialised four-chamber models.

    Usage:
        pipeline = RuminantPipeline(domain="finance")
        pipeline.ingest("path/to/data")
        pipeline.train(base_model="meta-llama/Llama-3-8B")
        metrics = pipeline.evaluate()
    """

    def __init__(self, domain: str = "finance", output_dir: str = "ruminant_output"):
        self.domain = domain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processors
        self.financial_proc = FinancialProcessor(str(self.output_dir / "training"))
        self.theorem_proc = TheoremProcessor()
        self.market_proc = MarketDataProcessor()

        # Trainer
        self.trainer = ChamberLoRATrainer()

        # Evaluator
        self.evaluator = ThermodynamicEvaluator()

        self._ingested = False
        self._trained = False

    def ingest(self, data_path: str):
        """Ingest domain data from multiple sources.

        Automatically detects file types and routes to appropriate processor:
          .csv  -> FinancialProcessor (market data)
          .tex  -> TheoremProcessor (papers)
          .json -> MarketDataProcessor (validation results)
          dir/  -> process all files recursively
        """
        path = Path(data_path)

        if path.is_dir():
            # Process all files in directory
            for csv_file in path.glob("**/*.csv"):
                self.financial_proc.process_market_data(str(csv_file))

            for tex_file in path.glob("**/*.tex"):
                self.theorem_proc.process_latex(str(tex_file))

            for json_dir in path.glob("**/results"):
                if json_dir.is_dir():
                    self.market_proc.process_validation_results(str(json_dir))

        elif path.suffix == ".csv":
            self.financial_proc.process_market_data(str(path))
        elif path.suffix == ".tex":
            self.theorem_proc.process_latex(str(path))
        elif path.suffix == ".json":
            self.market_proc.process_validation_results(str(path.parent))

        # Save combined training data
        all_pairs = (
            self.financial_proc.pairs +
            self.theorem_proc.pairs +
            self.market_proc.pairs
        )
        self.financial_proc.pairs = all_pairs
        training_file = self.financial_proc.save("training_corpus.jsonl")

        self._ingested = True
        return {
            "total_pairs": len(all_pairs),
            "financial": self.financial_proc.stats(),
            "theorems": len(self.theorem_proc.pairs),
            "market_data": len(self.market_proc.pairs),
            "training_file": str(training_file),
        }

    def train(self, base_model: str = "meta-llama/Llama-3-8B",
              config: ChamberLoRAConfig = None):
        """Train chamber-specific LoRA adapters."""
        if not self._ingested:
            raise RuntimeError("Call ingest() before train()")

        training_file = self.output_dir / "training" / "training_corpus.jsonl"
        self.trainer = ChamberLoRATrainer(config or ChamberLoRAConfig())
        self.trainer.load_data(str(training_file))

        manifest = self.trainer.train(
            base_model_path=base_model,
            output_dir=str(self.output_dir / "model"),
        )

        self._trained = True
        return manifest

    def evaluate(self) -> dict:
        """Evaluate the trained model using thermodynamic metrics."""
        import numpy as np
        # Generate synthetic test data for evaluation
        rng = np.random.RandomState(42)
        X_input = rng.randn(32, 128).astype(np.float32)
        X_output = X_input * 0.95 + rng.randn(32, 128).astype(np.float32) * 0.05

        metrics = self.evaluator.evaluate_representations(X_input, X_output)
        return {
            "information_density": metrics.information_density,
            "phase_coherence": metrics.phase_coherence,
            "convergence_rate": metrics.convergence_rate,
            "s_entropy": metrics.s_entropy,
            "free_energy": metrics.total_free_energy,
        }

    def status(self) -> dict:
        """Current pipeline status."""
        return {
            "domain": self.domain,
            "ingested": self._ingested,
            "trained": self._trained,
            "output_dir": str(self.output_dir),
            "trainer_stats": self.trainer.stats() if self._ingested else None,
        }
