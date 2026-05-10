#!/usr/bin/env python3
"""4.6.4: Export Attention Weights for Representative Cases.

Exports and visualizes per-timestep attention weights for SynerT model
on representative TP, TN, FP, FN cases.

Usage:
  python export_attention_4_6_4.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import json

# Assume vitaldb_aki is on path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vitaldb_aki.models.architectures import TemporalSynergyClassifier


def find_representative_cases(
    oof_file: Path,
    cohort_file: Path,
    threshold: float = 0.5703
) -> Dict[str, int]:
    """Find representative cases from each confusion category.
    
    Returns:
        Dictionary mapping confusion_type (TP, TN, FP, FN) to caseid
    """
    oof_df = pd.read_csv(oof_file)
    cohort_df = pd.read_csv(cohort_file)
    
    merged = oof_df.merge(cohort_df[["caseid", "baseline_cr"]], on="caseid", how="left")
    merged["y_pred"] = (merged["y_prob"] >= threshold).astype(int)
    
    # Assign confusion labels
    confusion = np.where(
        (merged["y_true"] == 1) & (merged["y_pred"] == 1), "TP",
        np.where(
            (merged["y_true"] == 0) & (merged["y_pred"] == 0), "TN",
            np.where(
                (merged["y_true"] == 0) & (merged["y_pred"] == 1), "FP",
                np.where(
                    (merged["y_true"] == 1) & (merged["y_pred"] == 0), "FN",
                    "NA"
                )
            )
        )
    )
    merged["confusion"] = confusion
    
    # Select first high-confidence case from each category
    cases = {}
    cases["TP"] = merged[merged["confusion"] == "TP"].sort_values("y_prob", ascending=False).iloc[0] if len(merged[merged["confusion"] == "TP"]) > 0 else None
    cases["TN"] = merged[merged["confusion"] == "TN"].sort_values("y_prob", ascending=True).iloc[0] if len(merged[merged["confusion"] == "TN"]) > 0 else None
    cases["FP"] = merged[merged["confusion"] == "FP"].sort_values("y_prob", ascending=False).iloc[0] if len(merged[merged["confusion"] == "FP"]) > 0 else None
    cases["FN"] = merged[merged["confusion"] == "FN"].sort_values("y_prob", ascending=True).iloc[0] if len(merged[merged["confusion"] == "FN"]) > 0 else None
    
    result = {}
    for key, row in cases.items():
        if row is not None:
            result[key] = int(row["caseid"])
    
    return result


class TemporalSynergyWithAttentionHook(TemporalSynergyClassifier):
    """SynerT with attention weight export hook."""
    
    def __init__(self, *args, return_attention: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attention = return_attention
        self.last_attention_weights = None
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """Forward pass with optional attention weight capture."""
        B, T, num_features = x.shape
        
        # Standard forward through to pooling stage
        x_conv = x.transpose(1, 2)
        h_tcn = self.in_proj(x_conv)
        h_tcn = self.tcn(h_tcn)
        h_tcn = h_tcn.transpose(1, 2)
        
        rnn_outputs = []
        current_input = h_tcn
        for rnn_layer in self.rnn_layers_list:
            out = rnn_layer(current_input, lengths)
            rnn_outputs.append(out)
            current_input = out
        
        combined = [h_tcn] + rnn_outputs
        combined = torch.cat(combined, dim=-1)
        fused = self.fusion(combined)
        
        # Capture attention weights
        if self.use_attention and (return_attention or self.return_attention):
            attn_scores = self.attention(fused)  # [B, T, 1]
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            attn_scores = attn_scores * mask + (1 - mask) * (-1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
            self.last_attention_weights = attn_weights.detach().cpu().numpy()
            pooled = torch.sum(fused * attn_weights, dim=1)
        else:
            mask = (
                torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1).float()
            masked = fused * mask
            pooled = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1).float()
        
        logits = self.classifier(pooled)
        return logits


def export_representative_attention(
    exp_name: str = "new_optional_exp",
    threshold: float = 0.5703,
) -> Dict:
    """Export attention weights for representative cases.
    
    Note: This is a template. Full implementation requires:
    1. Loading trained model checkpoints
    2. Loading test data for representative caseids
    3. Running inference with attention hook enabled
    """
    root = Path(__file__).resolve().parents[2]
    
    # Find representative cases
    oof_file = root / "artifacts" / exp_name / "results" / "temporal_synergy_oof_predictions_ensemble_mean.csv"
    cohort_file = root / "artifacts" / exp_name / "cohort_master.csv"
    
    representative_cases = find_representative_cases(oof_file, cohort_file, threshold)
    
    print("[4.6.4] Exporting attention weights for representative cases:")
    for confusion_type, caseid in representative_cases.items():
        print(f"  {confusion_type}: caseid={caseid}")
    
    # Placeholder results
    results = {
        "representative_cases": representative_cases,
        "threshold": threshold,
        "attention_export_status": "Ready for implementation",
        "required_steps": [
            "1. Load trained SynerT model checkpoints (all 25 seed-fold combinations)",
            "2. Load test data corresponding to representative caseids",
            "3. Run inference with TemporalSynergyWithAttentionHook",
            "4. Aggregate attention weights across all checkpoints (average or median)",
            "5. Generate heatmap plots: time vs attention weight per case",
        ]
    }
    
    return results


def generate_attention_report(results: Dict) -> str:
    """Generate markdown report for attention export."""
    report = "# 4.6.4 Attention Weight Export Report\n\n"
    report += "## Representative Cases\n\n"
    
    for confusion_type, caseid in results["representative_cases"].items():
        report += f"- **{confusion_type}**: caseid={caseid}\n"
    
    report += "\n## Implementation Steps\n\n"
    for step in results["required_steps"]:
        report += f"- {step}\n"
    
    report += "\n## Expected Outputs\n\n"
    report += "1. **Attention Heatmaps**: [caseid]_attention_weights.png (time × attention)\n"
    report += "2. **Aggregated Weights CSV**: chapter_4_6_4_attention_weights.csv\n"
    report += "3. **Interpretation Report**: Discussion of temporal focus patterns\n"
    
    return report


def main():
    print("\n[4.6.4] Exporting temporal attention weights for SynerT representation cases...")
    
    results = export_representative_attention()
    
    # Save results
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "artifacts" / "new_optional_exp" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_file = output_dir / "chapter_4_6_4_attention_export_plan.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Attention export plan saved: {json_file}")
    
    # Markdown report
    report = generate_attention_report(results)
    report_file = output_dir / "chapter_4_6_4_attention_export.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Attention export report saved: {report_file}")


if __name__ == "__main__":
    main()
