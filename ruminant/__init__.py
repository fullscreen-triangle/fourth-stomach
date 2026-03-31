"""
Ruminant: A Four-Chamber Domain Specialisation Framework

Unlike RAG (retrieve at inference) or standard fine-tuning (uniform adaptation),
Ruminant processes domain knowledge through four functionally distinct chambers:

  Chamber 1 (Rumen):     Circulation attention — dense token mixing
  Chamber 2 (Reticulum): Spectral attention — FFT harmonic coincidence detection
  Chamber 3 (Omasum):    Graph completion attention — fill gaps from confident to uncertain
  Chamber 4 (Abomasum):  Refinement attention — confidence-weighted output

The four-chamber forward pass is a contraction mapping. Rumination (iteration)
converges to a unique fixed point by the Banach fixed-point theorem.

Usage:
    from ruminant import RuminantPipeline

    pipeline = RuminantPipeline(domain="finance")
    pipeline.ingest("path/to/financial/data")
    pipeline.train(base_model="meta-llama/Llama-3-8B")
    result = pipeline.generate("What phase is the market in?")
"""

__version__ = "0.1.0"

from ruminant.pipeline import RuminantPipeline
