#!/usr/bin/env python3
"""
EZ Discrimination Analysis with MDT-Reviewed Ground Truth Labels

This script uses the corrected ground truth labels from ez_ground_truth.py
based on comprehensive MDT-style clinical review of:
- EEG/EMU FINDINGS
- CLINICAL MRI
- NEUROPSYCHOLOGY
- RESEARCH MRI

Includes options for:
1. Unilateral-only analysis (L vs R)
2. Including bilateral with dominant side
3. Full analysis including all bilateral cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Import ground truth labels
from ez_ground_truth import (
    EZ_GROUND_TRUTH,
    get_ez_label,
    get_unilateral_patients,
    get_bilateral_patients,
    get_left_ez_patients,
    get_right_ez_patients,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


class EZDiscriminatorMDT:
    """
    EZ discrimination using MDT-reviewed ground truth labels.

    Sign convention (interictal ASL hypoperfusion model)
    -----------------------------------------------------
    - Laterality Index (LI) = (L - R) / (L + R)
        Positive LI  → Left hemisphere has higher perfusion
        Negative LI  → Right hemisphere has higher perfusion
    - The epileptogenic zone (EZ) characteristically shows
      *interictal hypoperfusion*. Therefore:
        Left EZ  → Left hypoperfusion → R > L → NEGATIVE LI
        Right EZ → Right hypoperfusion → L > R → POSITIVE LI
    - This is counter-intuitive: a positive (L > R) LI suggests the
      *right* hemisphere harbours the EZ, not the left.

    The `evaluate_methods` routine accepts both interpretations
    by trying both `y_score` and `-y_score` and reporting the better
    AUC, with `flipped=True` indicating the inverse direction was used.

    Aggregation methods provided (suffix: which atlas signal they consume)
    --------------------------------------------------------------------
    - signed_weighted_li               : Σ(LI·|LI|) / Σ|LI|         all regions
    - temporal_signed_li               : Σ(LI·|LI|) / Σ|LI|         temporal only
    - mesial_temporal_li               : mean(LI)                   mesial-temporal only
    - mean_li                          : mean(LI)                   all regions
    - weighted_aai                     : Σ(AAI·|AAI|)/ Σ|AAI|       all regions  (S_weighted)
    - dominant_side_score              : (#L>0 - #R<0)/total        sign counting
    - cohen_d_directional              : standardised L vs R mean   parametric
    - anatomically_weighted            : weighted-average LI with    anatomical
                                          region-specific weights
                                          (mesial-temp 3.0, temp 2.0,
                                          frontal 2.0, default 0.5)
    - temporal_focused_score           : 0.6·mean_LI(mesial-temp) + temporal blend
                                          0.4·mean_LI(all-temp)

    Both `temporal_signed_li` and `temporal_focused_score` aggregate over
    temporal regions but use different math:
      * temporal_signed_li uses self-weighting (`Σ A·|A| / Σ |A|`)
      * temporal_focused_score uses a hierarchical blend of arithmetic
        means giving extra weight to mesial structures.
    """

    def __init__(self, results_dir="results_analysis"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("ez_analysis_mdt")
        self.output_dir.mkdir(exist_ok=True)

        # Anatomical regions
        self.mesial_temporal = [
            "entorhinal",
            "parahippocampal",
            "hippocampus",
            "amygdala",
        ]
        self.lateral_temporal = [
            "temporalpole",
            "fusiform",
            "inferiortemporal",
            "middletemporal",
            "superiortemporal",
            "transversetemporal",
            "bankssts",
        ]
        self.temporal_all = self.mesial_temporal + self.lateral_temporal

        # Frontal regions (used by anatomically_weighted with focus="frontal")
        self.frontal_regions = [
            "frontalpole",
            "medialorbitofrontal",
            "lateralorbitofrontal",
            "parsorbitalis",
            "parstriangularis",
            "parsopercularis",
            "rostralmiddlefrontal",
            "caudalmiddlefrontal",
            "superiorfrontal",
            "precentral",
            "paracentral",
        ]

        self.asymmetry_data = {}

    def load_data(self):
        """Load asymmetry data for all patients"""
        print("📋 Loading asymmetry data...")

        for patient_dir in sorted(self.results_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            csv_file = (
                patient_dir / f"{patient_id}_comprehensive_asymmetry_analysis.csv"
            )

            if csv_file.exists():
                df = pd.read_csv(csv_file)
                self.asymmetry_data[patient_id] = df

        print(f"   ✅ Loaded {len(self.asymmetry_data)} patients")

    # ==========================================================================
    # ASYMMETRY METHODS
    # ==========================================================================

    def signed_weighted_li(self, df):
        """Weighted LI preserving direction"""
        if "laterality_index" not in df.columns:
            return 0
        li = df["laterality_index"].values
        abs_li = np.abs(li)
        if abs_li.sum() == 0:
            return 0
        return (li * abs_li).sum() / abs_li.sum()

    def temporal_signed_li(self, df):
        """Temporal-focused weighted LI"""
        if "laterality_index" not in df.columns:
            return 0
        temporal_mask = (
            df["region"]
            .str.lower()
            .apply(lambda x: any(r in x for r in self.temporal_all))
        )
        if temporal_mask.sum() == 0:
            return self.signed_weighted_li(df)

        li = df.loc[temporal_mask, "laterality_index"].values
        abs_li = np.abs(li)
        if abs_li.sum() == 0:
            return 0
        return (li * abs_li).sum() / abs_li.sum()

    def mesial_temporal_li(self, df):
        """Mesial temporal focused LI"""
        if "laterality_index" not in df.columns:
            return 0
        mesial_mask = (
            df["region"]
            .str.lower()
            .apply(lambda x: any(r in x for r in self.mesial_temporal))
        )
        if mesial_mask.sum() == 0:
            return 0
        return df.loc[mesial_mask, "laterality_index"].mean()

    def mean_li(self, df):
        """Simple mean LI"""
        if "laterality_index" not in df.columns:
            return 0
        return df["laterality_index"].mean()

    def weighted_aai(self, df):
        """Original weighted AAI (magnitude only)"""
        if "absolute_asymmetry_index" not in df.columns:
            return 0
        aai = df["absolute_asymmetry_index"].values
        if np.abs(aai).sum() == 0:
            return 0
        return (aai * np.abs(aai)).sum() / np.abs(aai).sum()

    def dominant_side_score(self, df, threshold=0.05):
        """Score based on dominant asymmetry direction"""
        if "laterality_index" not in df.columns:
            return 0
        li = df["laterality_index"].values
        n_left = np.sum(li > threshold)
        n_right = np.sum(li < -threshold)
        n_total = n_left + n_right
        if n_total == 0:
            return 0
        return (n_left - n_right) / n_total

    def cohen_d_directional(self, df):
        """Cohen's d between left and right perfusion (directional, cross-ROI).

        Computes the standardised mean difference between the per-ROI
        ``left_mean_perfusion`` and ``right_mean_perfusion`` vectors.

        Bug fix history
        ---------------
        2026-05-04: Corrected the column lookup. The function previously
        read ``left_mean`` / ``right_mean``, which do not exist in the
        comprehensive-asymmetry CSVs produced by
        ``batch_analyze_all_patients.py``. The actual column names are
        ``left_mean_perfusion`` / ``right_mean_perfusion``. With the old
        names the guard returned 0 for every patient and the AUC
        degenerated to 0.500 (chance). Discovered during the
        2026-05-04 cohen_d / AUC refresh; see
        ``docs/superpowers/audits/2026-05-04-cohen_d-and-auc-refresh.md``.
        """
        if (
            "left_mean_perfusion" not in df.columns
            or "right_mean_perfusion" not in df.columns
        ):
            return 0
        left = df["left_mean_perfusion"].values
        right = df["right_mean_perfusion"].values
        pooled_std = np.sqrt((np.var(left) + np.var(right)) / 2)
        if pooled_std == 0:
            return 0
        return (np.mean(left) - np.mean(right)) / pooled_std

    def anatomically_weighted(self, df, focus="temporal"):
        """
        Anatomically-Weighted Asymmetry Index (AWAI).

        Weights regional laterality_index values by clinical relevance to
        epileptogenic zone. For temporal lobe epilepsy, mesial-temporal
        regions get the highest weight, lateral-temporal next, others lower.

        Implements the manuscript's `S_anatomical` aggregation:

            S_anatomical = Σ (w_i · A_i) / Σ w_i

        where `A_i = laterality_index_i` and `w_i ∈ {3, 2, 0.5}` for
        mesial-temporal / temporal / non-temporal under `focus="temporal"`.

        (Originally `calculate_anatomically_weighted_asymmetry` in
        `advanced_ez_discrimination.py`; ported during the 2026-05-04
        ez_discrimination consolidation.)

        Args:
            df: per-region asymmetry table with columns
                ["region", "laterality_index"].
            focus: "temporal" (TLE-tuned weights), "frontal" (FLE),
                or "" (uniform weight = 1.0).

        Returns:
            float — weighted asymmetry score
            (positive = left hyperperfusion, see class docstring on
            sign convention).
        """
        if "laterality_index" not in df.columns:
            return 0

        # Define weights based on focus
        if focus == "temporal":
            weight_map = {}
            for region in self.mesial_temporal:
                weight_map[region] = 3.0  # Highest weight for mesial temporal
            for region in self.temporal_all:
                if region not in weight_map:
                    weight_map[region] = 2.0  # High weight for lateral temporal
            default_weight = 0.5  # Low weight for non-temporal
        elif focus == "frontal":
            weight_map = {r: 2.0 for r in self.frontal_regions}
            default_weight = 0.5
        else:
            weight_map = {}
            default_weight = 1.0

        weighted_sum = 0.0
        total_weight = 0.0

        for _, row in df.iterrows():
            region = str(row["region"]).lower()
            # Match region to weight (first hit wins)
            weight = default_weight
            for key, w in weight_map.items():
                if key in region:
                    weight = w
                    break

            li = row["laterality_index"]
            weighted_sum += weight * li
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def temporal_focused_score(self, df):
        """
        Temporal-Focused Asymmetry Score (TFAS).

        Hierarchical blend of arithmetic means restricted to temporal
        regions, with extra weight on mesial-temporal structures:

            score = 0.6 * mean_LI(mesial_temporal)
                  + 0.4 * mean_LI(all_temporal)

        If no mesial-temporal regions are present, falls back to the
        all-temporal mean LI.

        This is *distinct* from `temporal_signed_li`, which applies
        magnitude self-weighting (Σ A·|A| / Σ |A|) within the temporal
        subset. Both are kept because they aggregate temporal-region
        evidence differently (means vs self-weighted means).

        (Originally `calculate_temporal_focused_score` in
        `advanced_ez_discrimination.py`; ported during the 2026-05-04
        ez_discrimination consolidation.)
        """
        if "laterality_index" not in df.columns:
            return 0

        temporal_mask = (
            df["region"]
            .str.lower()
            .apply(lambda x: any(r in x for r in self.temporal_all))
        )
        mesial_mask = (
            df["region"]
            .str.lower()
            .apply(lambda x: any(r in x for r in self.mesial_temporal))
        )

        all_temporal_mean = (
            df.loc[temporal_mask, "laterality_index"].mean()
            if temporal_mask.sum() > 0
            else 0
        )
        mesial_temporal_mean = (
            df.loc[mesial_mask, "laterality_index"].mean()
            if mesial_mask.sum() > 0
            else 0
        )

        if mesial_mask.sum() > 0:
            return 0.6 * mesial_temporal_mean + 0.4 * all_temporal_mean
        return all_temporal_mean

    # ==========================================================================
    # EVALUATION
    # ==========================================================================

    def evaluate_methods(self, include_bilateral="dominant"):
        """
        Evaluate all methods against ground truth

        Args:
            include_bilateral: How to handle bilateral cases
                - "exclude": Only use unilateral cases
                - "dominant": Include B-L as L, B-R as R, exclude pure B
                - "all_left": Treat all bilateral as L
                - "all_right": Treat all bilateral as R
        """
        print("\n" + "=" * 80)
        print(f"🔬 EZ DISCRIMINATION ANALYSIS (MDT Labels)")
        print(f"   Bilateral handling: {include_bilateral}")
        print("=" * 80)

        # Build evaluation dataset
        y_true = []
        patient_ids = []

        for patient_id in self.asymmetry_data.keys():
            if patient_id not in EZ_GROUND_TRUTH:
                continue

            ez = get_ez_label(patient_id, include_bilateral_as=include_bilateral)
            if ez is None:
                continue

            # Convert to binary: L=1, R=0
            y_true.append(1 if ez == "L" else 0)
            patient_ids.append(patient_id)

        n_left = sum(y_true)
        n_right = len(y_true) - n_left
        print(f"\n📊 Evaluating on {len(y_true)} patients")
        print(f"   Left EZ: {n_left}, Right EZ: {n_right}")

        if len(y_true) < 4:
            print("❌ Not enough patients for analysis")
            return None

        # Define methods
        methods = {
            "signed_weighted_li": self.signed_weighted_li,
            "temporal_signed_li": self.temporal_signed_li,
            "mesial_temporal_li": self.mesial_temporal_li,
            "mean_li": self.mean_li,
            "weighted_aai": self.weighted_aai,
            "dominant_side": self.dominant_side_score,
            "cohen_d": self.cohen_d_directional,
            "anatomically_weighted": self.anatomically_weighted,
            "temporal_focused_score": self.temporal_focused_score,
        }

        results = []
        all_scores = {}

        for method_name, method_func in methods.items():
            scores = []
            for pid in patient_ids:
                df = self.asymmetry_data[pid]
                score = method_func(df)
                scores.append(score)

            all_scores[method_name] = scores

            # Calculate metrics
            # Positive score = L > R perfusion
            # For interictal ASL: EZ shows hypoperfusion
            # So positive LI might indicate RIGHT EZ (contralateral hyperperfusion)
            # OR positive LI might indicate LEFT EZ is healthy (higher perfusion)
            # We need to determine the best interpretation empirically

            y_score = np.array(scores)
            y_true_arr = np.array(y_true)

            # Try both interpretations
            # Interpretation 1: Positive score → Left EZ (EZ has higher perfusion - ictal pattern)
            # Interpretation 2: Negative score → Left EZ (EZ has lower perfusion - interictal hypoperfusion)

            auc_direct = 0.5
            auc_flipped = 0.5

            try:
                fpr, tpr, _ = roc_curve(y_true_arr, y_score)
                auc_direct = auc(fpr, tpr)
            except:
                pass

            try:
                fpr, tpr, _ = roc_curve(y_true_arr, -y_score)
                auc_flipped = auc(fpr, tpr)
            except:
                pass

            # Use better interpretation
            if auc_flipped > auc_direct:
                best_auc = auc_flipped
                flip = True
                y_score_eval = -y_score
            else:
                best_auc = auc_direct
                flip = False
                y_score_eval = y_score

            # Find optimal threshold
            best_thresh = 0
            best_acc = 0
            for thresh in np.percentile(y_score_eval, np.arange(10, 91, 5)):
                preds = (y_score_eval > thresh).astype(int)
                acc = (preds == y_true_arr).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh

            # Calculate final metrics at optimal threshold
            preds = (y_score_eval > best_thresh).astype(int)

            tp = np.sum((preds == 1) & (y_true_arr == 1))
            tn = np.sum((preds == 0) & (y_true_arr == 0))
            fp = np.sum((preds == 1) & (y_true_arr == 0))
            fn = np.sum((preds == 0) & (y_true_arr == 1))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            correct = tp + tn

            results.append(
                {
                    "method": method_name,
                    "auc": best_auc,
                    "sensitivity": sens,
                    "specificity": spec,
                    "accuracy": best_acc,
                    "correct": correct,
                    "total": len(y_true),
                    "threshold": best_thresh,
                    "flipped": flip,
                }
            )

        # Sort by AUC
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("auc", ascending=False)

        # Print results
        print("\n" + "-" * 100)
        print(
            f"{'Method':<25} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'Correct':>10} {'Flip':>6}"
        )
        print("-" * 100)

        for _, row in results_df.iterrows():
            flip_str = "Yes" if row["flipped"] else "No"
            print(
                f"{row['method']:<25} {row['auc']:>8.3f} {row['sensitivity'] * 100:>7.1f}% "
                f"{row['specificity'] * 100:>7.1f}% {row['accuracy'] * 100:>7.1f}% "
                f"{int(row['correct']):>4}/{int(row['total']):<4} {flip_str:>6}"
            )

        # Patient-level predictions with best method
        best_method = results_df.iloc[0]["method"]
        best_flip = results_df.iloc[0]["flipped"]
        best_thresh = results_df.iloc[0]["threshold"]

        print(f"\n{'=' * 80}")
        print(f"📋 PATIENT-LEVEL PREDICTIONS")
        print(f"   Best Method: {best_method} (AUC={results_df.iloc[0]['auc']:.3f})")
        print(f"{'=' * 80}")

        print(
            f"\n{'Patient':<10} {'True EZ':<10} {'MDT Info':<15} {'Predicted':<12} {'Score':>10} {'Result':<8}"
        )
        print("-" * 80)

        for i, pid in enumerate(patient_ids):
            true_ez = "Left" if y_true[i] == 1 else "Right"
            ez_info = EZ_GROUND_TRUTH[pid]["ez"]
            score = all_scores[best_method][i]

            if best_flip:
                eval_score = -score
            else:
                eval_score = score

            pred = "Left" if eval_score > best_thresh else "Right"
            correct = (
                "✓"
                if (pred == "Left" and y_true[i] == 1)
                or (pred == "Right" and y_true[i] == 0)
                else "✗"
            )

            print(
                f"{pid:<10} {true_ez:<10} {ez_info:<15} {pred:<12} {score:>10.4f} {correct:<8}"
            )

        # Show misclassified
        print(f"\n⚠️  MISCLASSIFIED PATIENTS:")
        for i, pid in enumerate(patient_ids):
            score = all_scores[best_method][i]
            eval_score = -score if best_flip else score
            pred = 1 if eval_score > best_thresh else 0

            if pred != y_true[i]:
                true_ez = "Left" if y_true[i] == 1 else "Right"
                pred_ez = "Left" if pred == 1 else "Right"
                info = EZ_GROUND_TRUTH[pid]
                print(
                    f"   {pid}: True={true_ez} ({info['ez']}), Predicted={pred_ez}, Score={score:.4f}"
                )
                print(f"      Evidence: {info['evidence'][:70]}...")

        # Save results
        results_df.to_csv(
            self.output_dir / f"method_performance_{include_bilateral}.csv", index=False
        )

        return results_df, all_scores, patient_ids, y_true

    def run_all_analyses(self):
        """Run analysis with different bilateral handling strategies"""

        self.load_data()

        all_results = {}

        # Analysis 1: Unilateral only
        print("\n" + "🔹" * 40)
        print("ANALYSIS 1: UNILATERAL CASES ONLY")
        print("🔹" * 40)
        results1 = self.evaluate_methods(include_bilateral="exclude")
        if results1:
            all_results["unilateral_only"] = results1[0]

        # Analysis 2: Include bilateral with dominant side
        print("\n" + "🔹" * 40)
        print("ANALYSIS 2: INCLUDING BILATERAL (DOMINANT SIDE)")
        print("🔹" * 40)
        results2 = self.evaluate_methods(include_bilateral="dominant")
        if results2:
            all_results["with_bilateral_dominant"] = results2[0]

        # Analysis 3: Include all bilateral as uncertain (exclude pure B)
        # This is same as dominant since we only have B-L and B-R with clear dominance

        # Generate comparison plot
        self.plot_comparison(all_results)

        # Generate comprehensive report
        self.generate_report(all_results)

        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {self.output_dir}/")
        print("=" * 80)

        return all_results

    def plot_comparison(self, all_results):
        """Create comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: AUC comparison
        ax1 = axes[0]
        for analysis_name, results_df in all_results.items():
            methods = results_df["method"].values
            aucs = results_df["auc"].values
            x = np.arange(len(methods))
            ax1.barh(
                x + (0.2 if "bilateral" in analysis_name else -0.2),
                aucs,
                height=0.35,
                label=analysis_name.replace("_", " ").title(),
                alpha=0.8,
            )

        ax1.set_yticks(np.arange(len(methods)))
        ax1.set_yticklabels(methods)
        ax1.set_xlabel("AUC")
        ax1.set_title("Method Performance Comparison")
        ax1.legend()
        ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
        ax1.set_xlim(0, 1)

        # Plot 2: Accuracy comparison
        ax2 = axes[1]
        for analysis_name, results_df in all_results.items():
            methods = results_df["method"].values
            accs = results_df["accuracy"].values * 100
            x = np.arange(len(methods))
            ax2.barh(
                x + (0.2 if "bilateral" in analysis_name else -0.2),
                accs,
                height=0.35,
                label=analysis_name.replace("_", " ").title(),
                alpha=0.8,
            )

        ax2.set_yticks(np.arange(len(methods)))
        ax2.set_yticklabels(methods)
        ax2.set_xlabel("Accuracy (%)")
        ax2.set_title("Accuracy Comparison")
        ax2.legend()
        ax2.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="Chance")
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "method_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"\n📊 Comparison plot saved to {self.output_dir}/method_comparison.png")

    def generate_report(self, all_results):
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("EZ DISCRIMINATION ANALYSIS REPORT")
        report.append("Using MDT-Reviewed Ground Truth Labels")
        report.append("=" * 80)
        report.append("")

        # Ground truth summary
        report.append("GROUND TRUTH SUMMARY")
        report.append("-" * 40)
        left = get_left_ez_patients()
        right = get_right_ez_patients()
        bilateral = get_bilateral_patients()
        report.append(f"Left EZ: {len(left)} patients - {', '.join(sorted(left))}")
        report.append(f"Right EZ: {len(right)} patients - {', '.join(sorted(right))}")
        report.append(
            f"Bilateral: {len(bilateral)} patients - {', '.join(sorted(bilateral))}"
        )
        report.append("")

        # Results for each analysis
        for analysis_name, results_df in all_results.items():
            report.append("")
            report.append("=" * 80)
            report.append(f"ANALYSIS: {analysis_name.upper().replace('_', ' ')}")
            report.append("=" * 80)

            best = results_df.iloc[0]
            report.append(f"\nBest Method: {best['method']}")
            report.append(f"  AUC: {best['auc']:.3f}")
            report.append(f"  Sensitivity: {best['sensitivity'] * 100:.1f}%")
            report.append(f"  Specificity: {best['specificity'] * 100:.1f}%")
            report.append(f"  Accuracy: {best['accuracy'] * 100:.1f}%")
            report.append(f"  Correct: {int(best['correct'])}/{int(best['total'])}")

            report.append("\nAll Methods:")
            for _, row in results_df.iterrows():
                report.append(
                    f"  {row['method']}: AUC={row['auc']:.3f}, Acc={row['accuracy'] * 100:.1f}%"
                )

        report_text = "\n".join(report)

        with open(self.output_dir / "analysis_report.txt", "w") as f:
            f.write(report_text)

        print(f"\n📄 Report saved to {self.output_dir}/analysis_report.txt")


def main():
    analyzer = EZDiscriminatorMDT()
    results = analyzer.run_all_analyses()
    return results


if __name__ == "__main__":
    main()
