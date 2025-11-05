# analysis_script.py
This script performs the following steps:
1. Load the CSV data (named 'dataframe.csv').
2. Compute overall agreement metrics (Exact Match Rate, Within-One-Point Rate,
   Cohen's Kappa, Quadratic Weighted Kappa, Pearson Correlation, ICC, Cronbach's Alpha).
3. Compute the same metrics by subgroup (A, B, C), based on 'Student_ID'.
4. Assess inter-trial reliability by pivoting the AI scores across multiple trials
   for each response and computing:
   - Cronbach's Alpha across 7 trials
   - Test-retest correlation between Trial 1 and Trial 7

IMPORTANT: This script assumes that the CSV has at least the following columns:
   'nth_trial'   (the trial number, 1 through 7),
   'o1_Score'    (AI-assigned score),
   'Actual_Score' (human or ground-truth score),
   'Student_ID'   (indicating group A/B/C or similar).
Adjust column names accordingly if your dataset uses different headers.

Run:
    python analysis_script.py

# code_snippet_for_auto_scoring
line 143 is for OPENAI_API_KEY (sk-proj-...)
