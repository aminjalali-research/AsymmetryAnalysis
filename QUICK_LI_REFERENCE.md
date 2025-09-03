#  **Quick Reference: LI Map Visualization**

##  **Essential Commands**

### **Basic Viewing (FSLeyes)**
```bash
cd /home/amin/AsymmetryAnalysis
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
         laterality_maps/P013_laterality_index_map.nii.gz \
         -cm red-yellow -dr -0.4 0.4 -a 70 &
```

### **Significant Asymmetries Only**
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
         laterality_maps/P013_significant_asymmetry_mask.nii.gz \
         -cm red -dr 0 1 -a 80 &
```

### **Left vs Right Dominance**
```bash
fsleyes Dataset/P013/T1w_acpc_dc_restore.nii.gz \
         laterality_maps/P013_left_dominant_regions.nii.gz -cm red -a 70 \
         laterality_maps/P013_right_dominant_regions.nii.gz -cm blue -a 70 &
```

---

##  **Color Interpretation**

| Color | Meaning | Clinical Significance |
|-------|---------|---------------------|
| 🔴 **Red** | Right > Left | Rightward asymmetry |
| 🟡 **Yellow** | Left > Right | Leftward asymmetry |
| ⚫ **Dark** | Balanced | No asymmetry |
| 🔥 **Bright** | Strong asymmetry | Clinical attention |

---

##  **P013 Key Findings**

- **9/37 regions** significantly asymmetric (|LI| > 0.1)
- **26.4%** of brain voxels show asymmetry  
- **Right dominance pattern** in most regions
- **Strongest**: Entorhinal (-0.398), Frontalpole (-0.332)

---

##  **Troubleshooting**

| Problem | Solution |
|---------|----------|
| Map appears dark | Adjust `-dr` range (try -0.2 0.2) |
| Wrong colors | Use `-cm red-yellow` or `-cm blue-red` |
| No overlay visible | Check transparency `-a 70` |
| Misalignment | Verify same image space |

---

## 📁 **Generated Files**

 `P013_laterality_index_map.nii.gz` - Main visualization  
 `P013_significant_asymmetry_mask.nii.gz` - |LI| > 0.1  
 `P013_left_dominant_regions.nii.gz` - LI > 0.1  
 `P013_right_dominant_regions.nii.gz` - LI < -0.1  
 `P013_LI_histogram.png` - Statistical distribution  

---

##  **Clinical Thresholds**

- **|LI| > 0.4**: Strong (investigate artifacts)
- **|LI| > 0.2**: Moderate (clinically significant) 
- **|LI| > 0.1**: Mild (statistical significance)
- **|LI| ≤ 0.1**: Balanced/normal

*For complete instructions, see `LATERALITY_VISUALIZATION_GUIDE.md`*
