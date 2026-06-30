#!/usr/bin/env python3
"""
Ground Truth EZ Lateralization Labels — sourced from Rai (2026) MSc thesis Table 11.

AUTHORITATIVE SOURCE: Aikam Kaur Rai, "Investigating the Role of Arterial Spin
Labelling in Presurgical Evaluation of Epilepsy", Queen's University, June 2026,
Table 11 (Patient Demographics and Clinical Details, pp. 60-65). The "Clinical
Diagnosis" column there is the MDT-consensus final diagnosis and is the ground
truth used for all concordance analyses.

Patient mapping (this project's imaging IDs P013-P027 -> thesis patient #1-17) was
established by matching age + sex + lesion. 14 of the project's imaging patients map
to thesis patients. Per project policy (2026-06-22): FOLLOW THE THESIS — any patient
not present in the thesis is DISCARDED. P014 (37/F, R fusiform cortical thickening)
matches no thesis patient and is therefore EXCLUDED from the cohort. Thesis patients
#15 (30/M LTLE), #16 (31/F LTLE), #17 (27/F RTLE) have no imaging in this project's
Dataset/ and are not analyzable here.

Label scheme (ez):
    'L'  unilateral Left            'R'  unilateral Right
    'B'  bilateral (thesis BTLE)    'U'  unclear laterality (thesis "unclear")
Parietal cases carry ez='L' with note (left-lateralized, non-temporal).

Thesis cohort tally (n=17, P014 excluded): 13 unilateral, 2 bilateral (#6,#7),
2 unclear (#5,#8). Labels L=8, R=5, B=2, U=2:
  L = P013,P015,P017,P022,P025(parietal),P026,P028,P029
  R = P016,P023,P024,P027,P030
  B = P019,P020      U = P018,P021
Unilateral (L+R, n=13) is the honest Pipeline A discrimination cohort.
"""

# Patient explicitly excluded per "follow thesis / discard if absent" policy.
DISCARDED = {
    "P014": "Not present in Rai 2026 thesis Table 11 (37/F R fusiform cortical "
            "thickening matches no thesis patient). Excluded from cohort.",
}

# MDT ground truth — verbatim-sourced from thesis Table 11 (thesis_id = thesis patient #).
EZ_GROUND_TRUTH = {
    # ===== UNILATERAL LEFT =====
    "P013": {
        "ez": "L", "thesis_id": 1, "thesis_dx": "LTLE (fusiform FCD)", "confidence": "High",
        "evidence": "MRI: L fusiform gyrus FCD type I/III, L sylvian fissure PMG | "
                    "EMU: L posterior temporal spikes (interictal), 1 L posterior temporal "
                    "seizure (ictal); MEG: 46 L fusiform spikes | Neuropsych: non-localizing",
    },
    "P015": {
        "ez": "L", "thesis_id": 2, "thesis_dx": "Left ?posterior TLE", "confidence": "Moderate",
        "evidence": "MRI: L posterior temporal convexity cortical thickening, L vertical "
                    "rotation of hippocampus | EMU: non-lateralized rhythmic theta (interictal), "
                    "3 FBTCS w/ right head version, non-lateralized fast activity (ictal)",
    },
    "P017": {
        "ez": "L", "thesis_id": 4, "thesis_dx": "LTLE (mesial)", "confidence": "High",
        "evidence": "MRI: L temporal lobe, hippocampus & amygdala smaller | EMU: L temporal "
                    "sharps (interictal), 4 L temporal-onset seizures (ictal) | "
                    "Neuropsych: poor verbal memory (left temporal)",
    },
    "P022": {
        "ez": "L", "thesis_id": 9, "thesis_dx": "TLE, suspected left", "confidence": "Moderate",
        "evidence": "MRI: normal (research MRI normal) | 2021 EMU: L temporal theta, no seizures; "
                    "2024-05 possible R posterior temporal onset (unclear); 2024-08 L temporal "
                    "theta; high-density EEG: 9 L temporal spikes (L parahippocampal/hippocampus) | "
                    "Neuropsych: possible left temporal",
    },
    "P025": {
        "ez": "L", "thesis_id": 12, "thesis_dx": "LPLE (post-central neoplasm)", "confidence": "High",
        "evidence": "MRI: L post-central gyrus / parietal opercular low-grade neoplasm | "
                    "2024 EMU: L frontotemporal (interictal), no seizures | Neuropsych: impaired "
                    "language & verbal memory (left hemisphere)",
        "note": "PARIETAL (left-lateralized, non-temporal lobe epilepsy)",
    },
    "P026": {
        "ez": "L", "thesis_id": 13, "thesis_dx": "LTLE (anterior)", "confidence": "High",
        "evidence": "MRI: L anterior temporal pole/amygdala/hippocampus lesion ?glial neoplasm | "
                    "EEG: L temporal seizure; EMU: L temporal (interictal & ictal) | "
                    "Neuropsych: left temporal (poor verbal memory)",
    },
    # ===== UNILATERAL RIGHT =====
    "P016": {
        "ez": "R", "thesis_id": 3, "thesis_dx": "RTLE (mesial)", "confidence": "Moderate",
        "evidence": "MRI: R lateral frontal subcortical WM hyperintensities | EMU: right>left "
                    "temporal (interictal), R temporal (ictal) | Neuropsych: non-lateralizing "
                    "(bilateral temporal)",
    },
    "P023": {
        "ez": "R", "thesis_id": 10, "thesis_dx": "RTLE ?FCD (mesial)", "confidence": "Moderate",
        "evidence": "MRI: reported normal; on review possible R amygdala signal change | "
                    "EMU Jun 2024: R mid/posterior temporal (interictal), no seizures | "
                    "Neuropsych: right temporal (working memory, visuospatial)",
    },
    "P024": {
        "ez": "R", "thesis_id": 11, "thesis_dx": "RTLE (posterior)", "confidence": "High",
        "evidence": "POST-SURGICAL (included at neurologist request). MRI: R temporal partial "
                    "resection, R HS, R inferior frontal gliosis | 2024 EEG: 1 R posterior "
                    "temporal seizure; EMU: R posterior temporal (interictal), 1 R posterior "
                    "temporal + 1 unclear seizure (ictal) | Neuropsych: poor visual memory (R temporal)",
    },
    "P027": {
        "ez": "R", "thesis_id": 14, "thesis_dx": "RTLE", "confidence": "Moderate",
        "evidence": "MRI: R temporal lobe smaller, possible PMG superior R temporal/supramarginal "
                    "(research MRI normal) | 2022 EMU: generalized spike-wave, R temporal maximal "
                    "(interictal), 3 R temporal seizures (ictal); 2024 EMU: intermittent right>>left "
                    "temporal slowing, 3 R temporal seizures | Neuropsych: non-localizing",
    },
    # ===== BILATERAL (thesis BTLE) =====
    "P019": {
        "ez": "B", "thesis_id": 6, "thesis_dx": "BTLE", "confidence": "Low",
        "evidence": "MRI: incomplete L hippocampal inversion, possible L HS | 2022 EMU: left>right "
                    "temporal (interictal), 32 seizures L frontotemporal onset (ictal); 2024 EMU: "
                    "8 events bitemporal onset | Neuropsych: non-lateralizing",
    },
    "P020": {
        "ez": "B", "thesis_id": 7, "thesis_dx": "BTLE", "confidence": "Low",
        "evidence": "MRI: L temporal encephalocele | EMU: right>left temporal (interictal), "
                    "23 seizures (19 L mesial temporal, 4 R mesial temporal) | "
                    "Neuropsych: non-lateralizing (bitemporal involvement)",
    },
    # ===== ADDED 2026-06-22: thesis #15/#16/#17 (P028/P029/P030) =====
    # Not in clinical_spreadsheet.xlsx; processed perfusion sourced from
    # multimodal/Aikam/.../asl_analysis/. Mapping inferred from the exact
    # sequential pattern P013->#1 ... P027->#14 (P014 excluded) -> P028=#15,
    # P029=#16, P030=#17. All ages (30/31/27) fall in the FM_20_39 band.
    "P028": {
        "ez": "L", "thesis_id": 15, "thesis_dx": "LTLE", "confidence": "Moderate",
        "age": 30, "sex": "M",
        "evidence": "MRI: normal (research MRI normal) | EMU: L anterior temporal "
                    "(interictal), 1 L anterior temporal seizure (ictal) | "
                    "Neuropsych: non-localizing",
    },
    "P029": {
        "ez": "L", "thesis_id": 16, "thesis_dx": "LTLE", "confidence": "Moderate",
        "age": 31, "sex": "F",
        "evidence": "MRI: L incomplete hippocampal inversion, possible L collateral "
                    "sulcus FCD (only hippocampal inversion confirmed on research MRI) | "
                    "EEG: L anterior temporal (interictal); 2024/2025 EMU L anterior "
                    "temporal, 4 L temporal seizures | Neuropsych: left temporal",
    },
    "P030": {
        "ez": "R", "thesis_id": 17, "thesis_dx": "RTLE", "confidence": "Moderate",
        "age": 27, "sex": "F",
        "evidence": "MRI: R temporal lobe smaller, possible PMG superior R temporal/"
                    "supramarginal (research MRI normal) | 2022 EMU generalized spike-wave "
                    "R temporal maximal, 3 R temporal seizures; 2024 EMU intermittent "
                    "right>>left temporal slowing, 3 R temporal seizures | non-localizing",
    },
    # ===== UNCLEAR LATERALITY (thesis "unclear") =====
    "P018": {
        "ez": "U", "thesis_id": 5, "thesis_dx": "TLE; unclear laterality (L imaging, R EEG); likely bilateral",
        "confidence": "Low",
        "evidence": "MRI: bilateral hippocampal atrophy (L>R, possible L HS) | EMU: R temporal "
                    "(interictal), R temporal seizures (ictal) | Neuropsych: non-localizing",
    },
    "P021": {
        "ez": "U", "thesis_id": 8, "thesis_dx": "TLE; unclear laterality (R vs BTLE)",
        "confidence": "Low",
        "evidence": "MRI: no relevant findings (incl. research MRI) | EMU: right>left temporal "
                    "(interictal), 6 events (5 R temporal, 1 possible L) (ictal) | "
                    "Neuropsych: non-localizing",
    },
}


def get_ez_label(patient_id, include_bilateral_as=None):
    """
    Get EZ lateralization label for a patient.

    Args:
        patient_id: Patient ID (e.g., 'P013'). Returns None if discarded/unknown.
        include_bilateral_as: How to treat non-unilateral cases ('B' bilateral, 'U' unclear):
            - None      : return the raw label ('L','R','B','U')
            - 'exclude' : return None for any non-unilateral case (B or U)
            - 'left'    : treat bilateral/unclear as Left
            - 'right'   : treat bilateral/unclear as Right

    Returns:
        str 'L'/'R'/'B'/'U', or None.
    """
    if patient_id not in EZ_GROUND_TRUTH:
        return None

    ez = EZ_GROUND_TRUTH[patient_id]["ez"]
    if include_bilateral_as is None or ez in ("L", "R"):
        return ez
    if include_bilateral_as == "exclude":
        return None
    if include_bilateral_as == "left":
        return "L"
    if include_bilateral_as == "right":
        return "R"
    return ez


def get_unilateral_patients():
    """Patients with a clear unilateral EZ (ez in L/R)."""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] in ("L", "R")]


def get_bilateral_patients():
    """Patients with bilateral EZ (thesis BTLE; ez == 'B')."""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "B"]


def get_unclear_patients():
    """Patients with unclear laterality (thesis 'unclear'; ez == 'U')."""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "U"]


def get_left_ez_patients():
    """Patients with Left EZ."""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "L"]


def get_right_ez_patients():
    """Patients with Right EZ."""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "R"]


def print_summary():
    """Print summary of ground truth labels."""
    print("\n" + "=" * 80)
    print("EZ GROUND TRUTH LABELS  (source: Rai 2026 thesis, Table 11)")
    print("=" * 80)

    left, right = get_left_ez_patients(), get_right_ez_patients()
    bilat, unclear = get_bilateral_patients(), get_unclear_patients()

    def _line(pid):
        i = EZ_GROUND_TRUTH[pid]
        note = f"  [{i['note']}]" if "note" in i else ""
        print(f"   {pid} (thesis #{i['thesis_id']:>2}): {i['thesis_dx']} "
              f"— {i['confidence']} conf{note}")

    print(f"\nLEFT  ({len(left)}):")
    [_line(p) for p in sorted(left)]
    print(f"\nRIGHT ({len(right)}):")
    [_line(p) for p in sorted(right)]
    print(f"\nBILATERAL ({len(bilat)}):")
    [_line(p) for p in sorted(bilat)]
    print(f"\nUNCLEAR ({len(unclear)}):")
    [_line(p) for p in sorted(unclear)]

    print("\n" + "-" * 80)
    print(f"DISCARDED (not in thesis): {', '.join(DISCARDED) or 'none'}")
    for pid, why in DISCARDED.items():
        print(f"   {pid}: {why}")
    print("-" * 80)
    print(f"TOTAL analyzable: {len(EZ_GROUND_TRUTH)} "
          f"(L={len(left)}, R={len(right)}, B={len(bilat)}, U={len(unclear)})")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
