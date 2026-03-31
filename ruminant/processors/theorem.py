"""
Theorem Processor

Extracts (theorem, proof) and (definition, explanation) pairs from LaTeX papers.
These pairs teach the model to REASON about the framework, not just retrieve facts.
"""

import re
from pathlib import Path
from ruminant.processors.financial import TrainingPair


class TheoremProcessor:
    """Extracts structured mathematical knowledge from LaTeX sources."""

    def __init__(self):
        self.pairs: list[TrainingPair] = []

    def process_latex(self, tex_path: str) -> list[TrainingPair]:
        """Extract theorem/proof and definition/explanation pairs from a .tex file."""
        text = Path(tex_path).read_text(encoding='utf-8', errors='ignore')
        pairs = []

        # Extract theorem-proof pairs
        theorem_pattern = r'\\begin\{theorem\}(.*?)\\end\{theorem\}'
        proof_pattern = r'\\begin\{proof\}(.*?)\\end\{proof\}'

        theorems = re.findall(theorem_pattern, text, re.DOTALL)
        proofs = re.findall(proof_pattern, text, re.DOTALL)

        for i, thm in enumerate(theorems):
            thm_clean = self._clean_latex(thm)
            if i < len(proofs):
                proof_clean = self._clean_latex(proofs[i])
                pairs.append(TrainingPair(
                    context=f"Prove the following theorem:\n{thm_clean}",
                    completion=f"Proof:\n{proof_clean}",
                    source=f"theorem:{Path(tex_path).stem}",
                    chamber_hint="abomasum",  # backward trajectory reasoning
                ))

        # Extract definitions
        def_pattern = r'\\begin\{definition\}(.*?)\\end\{definition\}'
        definitions = re.findall(def_pattern, text, re.DOTALL)
        for defn in definitions:
            defn_clean = self._clean_latex(defn)
            pairs.append(TrainingPair(
                context=f"Explain the following definition:\n{defn_clean}",
                completion=f"This definition establishes: {defn_clean[:200]}...",
                source=f"definition:{Path(tex_path).stem}",
                chamber_hint="rumen",  # broad understanding
            ))

        # Extract remarks (often contain key insights)
        remark_pattern = r'\\begin\{remark\}(.*?)\\end\{remark\}'
        remarks = re.findall(remark_pattern, text, re.DOTALL)
        for rmk in remarks:
            rmk_clean = self._clean_latex(rmk)
            if len(rmk_clean) > 50:
                pairs.append(TrainingPair(
                    context=f"What is the significance of:\n{rmk_clean[:300]}",
                    completion=rmk_clean,
                    source=f"remark:{Path(tex_path).stem}",
                    chamber_hint="reticulum",  # pattern recognition
                ))

        self.pairs.extend(pairs)
        return pairs

    def _clean_latex(self, text: str) -> str:
        """Remove LaTeX commands, keeping readable mathematical content."""
        text = re.sub(r'\\label\{[^}]*\}', '', text)
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        text = re.sub(r'\\ref\{[^}]*\}', '[ref]', text)
        text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\\\\s*', '\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_all_papers(self, papers_dir: str) -> list[TrainingPair]:
        """Process all .tex files in a directory."""
        papers_path = Path(papers_dir)
        all_pairs = []
        for tex_file in papers_path.glob("**/*.tex"):
            pairs = self.process_latex(str(tex_file))
            all_pairs.extend(pairs)
        return all_pairs
