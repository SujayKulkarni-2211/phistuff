Predictive Modelling of Healthcare Burden-Infrastructure Gaps Across Indian States
==================================================================================

CONTENTS
--------
  main.tex          Complete IEEE-format research paper (LaTeX source)
  experiment.py     Fully reproducible experiment script
  figures/          6 publication-quality PNG figures (300 DPI)
  results.json      All numeric results referenced in the paper
  README.txt        This file

HOW TO REPRODUCE THE EXPERIMENT
--------------------------------
1. Install dependencies:
     pip install kagglehub pandas numpy scikit-learn matplotlib seaborn scipy joblib

2. Set Kaggle credentials:
     export KAGGLE_USERNAME=
     export KAGGLE_KEY=

  

3. Run the experiment from the project directory:
     python experiment.py

   The script will:
   - Download all four Kaggle datasets automatically via kagglehub (~200 KB total)
   - Build the panel dataset (26 states × 2 time points)
   - Compute Burden-Infrastructure Gap Index
   - Train and LOO-CV evaluate three ML models
   - Generate all 6 figures in figures/
   - Write results.json with all numbers referenced in the paper

   Runtime: ~60 seconds on a standard laptop.

HOW TO COMPILE THE PAPER
--------------------------
Requires: a LaTeX distribution (TeX Live, MikTeX, or MacTeX) with IEEEtran class.

Method 1 — pdflatex (recommended):
    pdflatex main.tex
    pdflatex main.tex    # second pass for cross-references
    # Output: main.pdf

Method 2 — latexmk (auto-handles passes):
    latexmk -pdf main.tex
    # Output: main.pdf

Method 3 — Overleaf:
    Upload main.tex and the figures/ folder.
    Set compiler to pdfLaTeX.
    Click Compile.

The paper uses only standard IEEE packages (no external bibliography file required
— references are inline in the BibTeX array format).

FIGURES (in order)
------------------
  fig1_burden_heatmap.png     State-wise health burden index (NFHS-3 and NFHS-4)
  fig2_infra_heatmap.png      State-wise infrastructure index (RHS 2005 and 2019)
  fig3_gap_scatter.png        Burden vs infrastructure scatter by state (2015)
  fig4_model_comparison.png   LOO-CV MAE and R² for all three models
  fig5_feature_importance.png Feature importances for best model (Linear Regression)
  fig6_projection_2030.png    Top-10 at-risk states projected to 2030 with 90% CI

KEY METHODOLOGICAL DECISIONS (see experiment.py docstring for full rationale)
-----------------------------------------------------------------------------
1. Small UTs excluded (Lakshadweep etc.) — per-capita outliers distort normalisation
2. Z-normalisation within each time period — avoids secular trend leakage into LR
3. ML predicts 2015 gap from 2005 raw indicators — genuine predictive formulation
4. Leave-One-Out CV — appropriate for n=19 states, prevents optimistic hold-out metrics

DATASETS USED
-------------
  Rural Health Statistics 2005/2019/2020
  kaggle: vineethakkinapalli/rural-health-statistics-india2005-2019-2020

  NFHS-4 (National Family Health Survey, Round 4, 2015-16)
  kaggle: awadhi123/national-family-health-survey4

  NFHS-3 (National Family Health Survey, Round 3, 2005-06)
  kaggle: awadhi123/national-family-health-survey3-web-crawled
