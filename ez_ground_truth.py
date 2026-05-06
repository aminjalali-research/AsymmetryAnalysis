#!/usr/bin/env python3
"""
Ground Truth EZ Lateralization Labels

Based on comprehensive MDT-style clinical review of:
- EEG/EMU FINDINGS
- CLINICAL MRI
- NEUROPSYCHOLOGY
- RESEARCH MRI
- SURGICAL PLAN AFTER MDT

Labels: 'L' (Left), 'R' (Right), 'B' (Bilateral), 'B-L' (Bilateral L-dominant),
        'B-R' (Bilateral R-dominant), 'Unknown', 'Parietal' (non-temporal)
"""

# MDT-reviewed ground truth labels
EZ_GROUND_TRUTH = {
    # ===== UNILATERAL LEFT =====
    "P013": {
        "ez": "L",
        "confidence": "High",
        "evidence": "EEG: L mesial temporal | MRI: L fusiform FCD, L amygdala 17% smaller | sEEG planned for L temporal",
    },
    "P015": {
        "ez": "L",
        "confidence": "Moderate",
        "evidence": "EEG: Non-lateralized but 20 seizures | MRI: L posterior temporal FCD | Research: L posterior temporal thickening",
    },
    "P017": {
        "ez": "L",
        "confidence": "High",
        "evidence": "EEG: L frontotemporal slowing & sharps | MRI: L hippocampal atrophy (2nd MDT) | Plan: sEEG → L anterior temporal resection",
    },
    "P019": {
        "ez": "L",
        "confidence": "Moderate",
        "evidence": "EEG: 32 events L frontotemporal (2022), 8 events bilateral R preference (2024) | MRI: L hippocampal signal alteration",
    },
    "P020": {
        "ez": "L",
        "confidence": "High",
        "evidence": "EEG: 19/23 L anterior temporal onset, 4 R onset (2024) | MRI: Negative | Abundant L>R epileptiform activity",
    },
    "P025": {
        "ez": "L",
        "confidence": "High",
        "evidence": "MRI: L postcentral gyrus/parietal operculum lesion | MELDgraph: L postcentral 50%",
        "note": "PARIETAL - not temporal lobe epilepsy",
    },
    "P026": {
        "ez": "L",
        "confidence": "High",
        "evidence": "EEG: L temporal theta-delta slowing | MRI: L anterior temporal expansile lesion | Neuropsych: L temporal deficits",
    },
    # ===== UNILATERAL RIGHT =====
    "P014": {
        "ez": "R",
        "confidence": "High",
        "evidence": "EEG: R frontotemporal sharps and slowing | MRI: R fusiform cortical thickening | Research: R fusiform blurring",
    },
    "P016": {
        "ez": "R",
        "confidence": "Moderate",
        "evidence": "EEG: Multiple R posterior temporal seizures, few L temporal | MRI: Negative | Not eligible for surgery → VNS",
    },
    "P023": {
        "ez": "R",
        "confidence": "High",
        "evidence": "EEG: 7 events R anterior temporal, 'very clearly R' | MRI: R amygdala FCD | Neuropsych: R hippocampal dysfunction",
    },
    "P024": {
        "ez": "R",
        "confidence": "High",
        "evidence": "EEG: R posterior temporal localization | MRI: R mesial temporal sclerosis | AIDHS: R HS 100% confidence",
    },
    "P027": {
        "ez": "R",
        "confidence": "Moderate",
        "evidence": "EEG: R temporal sharps, R mid-posterior localization | MRI: R hippocampus flattening",
    },
    # ===== BILATERAL =====
    "P018": {
        "ez": "B-L",  # Bilateral, Left dominant
        "confidence": "Moderate",
        "evidence": "EEG: R temporal or bilateral onset | MRI: Bilateral HS (L>R atrophy) | AIDHS: L or Bilateral HS (L more pathological)",
    },
    "P021": {
        "ez": "B-R",  # Bilateral, Right dominant
        "confidence": "Low",
        "evidence": "EEG: Bitemporal, most from R temporal | MRI: Negative | Research: Bilateral hippocampal atrophy",
    },
    "P022": {
        "ez": "B",  # Bilateral, no clear dominance
        "confidence": "Low",
        "evidence": "EEG: L temporal slowing (2021) vs R posterior temporal ictal (2024) | MRI: Bilateral hippocampal atrophy",
    },
}


def get_ez_label(patient_id, include_bilateral_as=None):
    """
    Get EZ lateralization label for a patient.

    Args:
        patient_id: Patient ID (e.g., 'P013')
        include_bilateral_as: How to treat bilateral cases:
            - None: Return actual label ('B', 'B-L', 'B-R')
            - 'dominant': Return dominant side for B-L/B-R, exclude pure B
            - 'exclude': Return None for all bilateral
            - 'left': Treat all bilateral as Left
            - 'right': Treat all bilateral as Right

    Returns:
        str: 'L', 'R', 'B', 'B-L', 'B-R', or None
    """
    if patient_id not in EZ_GROUND_TRUTH:
        return None

    ez = EZ_GROUND_TRUTH[patient_id]["ez"]

    if include_bilateral_as is None:
        return ez

    if ez in ["L", "R"]:
        return ez

    # Handle bilateral cases
    if include_bilateral_as == "dominant":
        if ez == "B-L":
            return "L"
        elif ez == "B-R":
            return "R"
        else:
            return None  # Pure bilateral excluded
    elif include_bilateral_as == "exclude":
        return None
    elif include_bilateral_as == "left":
        return "L"
    elif include_bilateral_as == "right":
        return "R"

    return ez


def get_unilateral_patients():
    """Get list of patients with clear unilateral EZ"""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] in ["L", "R"]]


def get_bilateral_patients():
    """Get list of patients with bilateral involvement"""
    return [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"].startswith("B")]


def get_left_ez_patients(include_bilateral_dominant=False):
    """Get patients with Left EZ (optionally including B-L)"""
    patients = [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "L"]
    if include_bilateral_dominant:
        patients += [
            pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "B-L"
        ]
    return patients


def get_right_ez_patients(include_bilateral_dominant=False):
    """Get patients with Right EZ (optionally including B-R)"""
    patients = [pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "R"]
    if include_bilateral_dominant:
        patients += [
            pid for pid, info in EZ_GROUND_TRUTH.items() if info["ez"] == "B-R"
        ]
    return patients


def print_summary():
    """Print summary of ground truth labels"""
    print("\n" + "=" * 80)
    print("EZ GROUND TRUTH LABELS (MDT-Reviewed)")
    print("=" * 80)

    left = get_left_ez_patients()
    right = get_right_ez_patients()
    bilateral = get_bilateral_patients()

    print(f"\n🔵 LEFT EZ ({len(left)} patients):")
    for pid in sorted(left):
        info = EZ_GROUND_TRUTH[pid]
        note = f" ⚠️ {info.get('note', '')}" if "note" in info else ""
        print(f"   {pid}: {info['confidence']} confidence{note}")
        print(f"      {info['evidence'][:80]}...")

    print(f"\n🔴 RIGHT EZ ({len(right)} patients):")
    for pid in sorted(right):
        info = EZ_GROUND_TRUTH[pid]
        print(f"   {pid}: {info['confidence']} confidence")
        print(f"      {info['evidence'][:80]}...")

    print(f"\n🟡 BILATERAL ({len(bilateral)} patients):")
    for pid in sorted(bilateral):
        info = EZ_GROUND_TRUTH[pid]
        dom = (
            "(L-dominant)"
            if info["ez"] == "B-L"
            else "(R-dominant)"
            if info["ez"] == "B-R"
            else "(no dominance)"
        )
        print(f"   {pid}: {info['ez']} {dom} - {info['confidence']} confidence")
        print(f"      {info['evidence'][:80]}...")

    print("\n" + "=" * 80)
    print(f"TOTAL: {len(EZ_GROUND_TRUTH)} patients with labels")
    print(f"  Unilateral: {len(left) + len(right)} (L={len(left)}, R={len(right)})")
    print(f"  Bilateral: {len(bilateral)}")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
