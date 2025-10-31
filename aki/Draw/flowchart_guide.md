# VitalDB AKI Prediction Flowchart Guide

## Complete Flowchart Structure for Research Paper

### Layout: Vertical Flow (Top to Bottom)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT DATA (Parallelogram)                               │
│    VitalDB Database                                          │
│    • Cases Data (demographics, surgery info)                │
│    • Lab Data (creatinine levels)                           │
│    Color: #2E86AB                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. DATA PREPROCESSING (Rounded Rectangle)                   │
│    • Merge datasets                                          │
│    • Feature engineering                                     │
│    • AKI labeling (KDIGO Stage I)                           │
│    • Remove categoricals                                     │
│    Color: #06A77D                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. DATA QUALITY HANDLING (Rounded Rectangle)                │
│    • Missing value imputation                               │
│    • Train/Test split (80/20)                               │
│    • Handle class imbalance                                  │
│    Color: #06A77D                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
              ┌──────┴──────┐
              │             │
┌─────────────▼──┐    ┌────▼─────────────┐
│ Scaled Data    │    │ Imputed Data      │
│ (Standardized) │    │ (Mean imputation) │
│ Color: #3498DB │    │ Color: #F18F01    │
└─────────────┬──┘    └────┬──────────────┘
              │             │
              └──────┬──────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. ML MODEL TRAINING (Hexagon)                              │
│    Hyperparameter Tuning (GridSearchCV)                     │
│    • Logistic Regression                                     │
│    • Random Forest                                           │
│    • XGBoost                                                 │
│    • SVM                                                     │
│    Color: #6A4C93                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. MODEL EVALUATION (Rounded Rectangle)                     │
│    Performance Metrics                                       │
│    • ROC-AUC                                                 │
│    • AUPRC                                                   │
│    • Precision, Recall, F1-Score                            │
│    Color: #C73E1D                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 6. BEST MODEL SELECTION (Diamond)                          │
│    Rank by ROC-AUC                                           │
│    Color: #27AE60                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 7. MODEL INTERPRETABILITY (Rounded Rectangle)               │
│    SHAP Explanations                                         │
│    • Feature importance                                      │
│    • Individual predictions                                  │
│    Color: #8E44AD                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 8. OUTPUT (Oval)                                            │
│    Clinical Decision Support                                 │
│    AKI Risk Prediction                                        │
│    Color: #FFB627                                           │
└─────────────────────────────────────────────────────────────┘
```

## Color Code Reference

### Primary Colors
| Element | Color Code | Color Name | Use Case |
|---------|-----------|------------|----------|
| Input Data | `#2E86AB` | Ocean Blue | Starting point |
| Preprocessing | `#06A77D` | Medical Teal | Data processing |
| Scaled Data | `#3498DB` | Bright Blue | Standardized data |
| Imputed Data | `#F18F01` | Warm Orange | Tree-based models |
| ML Training | `#6A4C93` | Deep Purple | Model training |
| Evaluation | `#C73E1D` | Clinical Red | Validation |
| Best Model | `#27AE60` | Green | Success/output |
| Interpretability | `#8E44AD` | Purple | SHAP analysis |
| Final Output | `#FFB627` | Golden Yellow | Decision support |

### Supporting Colors
| Element | Color Code | Use Case |
|---------|-----------|----------|
| Background | `#FFFFFF` | White background |
| Text | `#2C3E50` | Dark blue-gray text |
| Border | `#34495E` | Medium gray borders |
| Arrows | `#BDC3C7` | Light gray connectors |

## Draw.io Implementation Instructions

### Step-by-Step Setup

1. **Page Setup**
   - Page size: A4 Landscape (297 x 210 mm)
   - Grid: 10mm
   - Snap to grid: Enabled

2. **Create Shapes**
   - Use rounded rectangles for processes (corner radius: 10-15px)
   - Use parallelograms for data input/output
   - Use diamonds for decision points
   - Use hexagons for ML model training
   - Use ovals for start/end

3. **Color Each Element**
   - Select shape → Fill color → Enter hex code
   - Text color: White (#FFFFFF) for colored boxes
   - Text color: #2C3E50 for diamonds and light backgrounds

4. **Connect Elements**
   - Use straight arrows (→)
   - Arrow color: #34495E
   - Line width: 2pt
   - Fill arrowhead

5. **Text Formatting**
   - Font: Arial or Helvetica
   - Size: 12pt for main text, 10pt for sub-items
   - Bold for main titles
   - Bullet points for details

6. **Professional Touches**
   - Add shadows: 3px offset, 30% opacity
   - Align all elements center
   - Even spacing: 40-60px vertical
   - Group related elements

## Visual Hierarchy

### Large Elements (Most Important)
- Input Data
- ML Model Training
- Output

### Medium Elements
- Preprocessing steps
- Evaluation metrics

### Small Elements (Details)
- Individual metrics
- Model types

## Export Settings

For research paper:
- Resolution: 300 DPI
- Format: PNG or PDF
- Background: Transparent or white
- Border: None

## Pro Tips

1. **Consistency**: Use same color for related elements
2. **Alignment**: Center-align all elements vertically
3. **Spacing**: Keep consistent gaps between elements
4. **Readability**: Ensure text contrasts with background
5. **Simplicity**: High-level = fewer details, cleaner look

## Common Issues & Solutions

**Issue**: Colors look different in draw.io vs exported image
**Solution**: Use hex codes directly, not color picker

**Issue**: Text not readable
**Solution**: Use white text on dark colors, dark text on light

**Issue**: Flowchart too crowded
**Solution**: Reduce detail, focus on main path

**Issue**: Export quality poor
**Solution**: Increase DPI to 300, use PDF format

