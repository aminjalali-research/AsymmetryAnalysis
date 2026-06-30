import sys, os
from pathlib import Path
BASE = Path('.')
mode = sys.argv[1]  # 'val' | 'full' | 'rest'
bands = ['FM_20_39', 'FM_40_59', 'FM_60_79']
ctrl = {b: sorted(str(p) for p in (BASE/'DatasetControls'/b).iterdir() if p.is_dir())
        for b in bands}
pat_band = {'FM_20_39': ['P013','P015','P016','P017','P019','P020','P026','P028','P029','P030'],
            'FM_40_59': ['P018','P021','P022','P023','P024','P027'],
            'FM_60_79': ['P025']}

def ctrl_job(d):
    n = Path(d).name
    return (f"{d}/MNINonLinear/T1w_restore_brain.nii.gz",
            f"{d}/{n}_MNISpace_perfusion_calib_upsampled.nii.gz",
            f"symreg/sym_perf/{n}",
            f"{d}/aparc+aseg.nii.gz")

def pat_job(p):
    return (f"symreg/patient_brains/{p}_brain.nii.gz",
            f"Dataset/{p}/{p}_perfusion_calib_resampled_to_T1w.nii.gz",
            f"symreg/sym_perf/{p}",
            f"Dataset/{p}/aparc+aseg.nii.gz")

jobs = []
if mode == 'val':
    jobs += [ctrl_job(d) for d in ctrl['FM_20_39'][:20]]
    jobs += [pat_job(p) for p in pat_band['FM_20_39']]
else:
    for b in bands:
        jobs += [ctrl_job(d) for d in ctrl[b]]
    for b in bands:
        jobs += [pat_job(p) for p in pat_band[b]]

for t1, perf, out, aparc in jobs:
    if mode == 'rest' and os.path.exists(f"{out}_perf_sym.nii.gz") and os.path.exists(f"{out}_aparc_sym.nii.gz"):
        continue
    if not os.path.exists(t1): sys.stderr.write(f"MISSING T1 {t1}\n"); continue
    if not os.path.exists(perf): sys.stderr.write(f"MISSING PERF {perf}\n"); continue
    if not os.path.exists(aparc): sys.stderr.write(f"MISSING APARC {aparc}\n"); continue
    print(f"{t1}\t{perf}\t{out}\t{aparc}")
