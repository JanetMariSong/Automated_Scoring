import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

"""
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
"""

def intraclass_correlation(ra1, ra2):
    """
    Compute a two-way random-effects, single-measure Intraclass Correlation Coefficient (ICC).
    Reference formula is based on ANOVA components (Shrout & Fleiss, 1979).
    ra1, ra2 : arrays/lists of scores from two raters for the same items.
    """
    # Number of items (responses)
    n = len(ra1)
    # Number of raters
    k = 2

    # Combine into a single array for ANOVA components
    y_true = np.array(ra1)
    y_pred = np.array(ra2)
    grand_mean = np.concatenate((y_true, y_pred)).mean()

    # Means per item (averaging across both raters)
    mean_per_item = (y_true + y_pred) / 2.0

    # Means per rater
    mean_rater1 = y_true.mean()
    mean_rater2 = y_pred.mean()

    # Sum-of-squares between items
    ss_between_items = k * np.sum((mean_per_item - grand_mean) ** 2)

    # Sum-of-squares between raters
    ss_between_raters = n * ((mean_rater1 - grand_mean) ** 2 + (mean_rater2 - grand_mean) ** 2)

    # Total sum-of-squares
    ss_total = np.sum((y_true - grand_mean) ** 2) + np.sum((y_pred - grand_mean) ** 2)

    # Residual sum-of-squares
    ss_residual = ss_total - ss_between_items - ss_between_raters

    # Mean squares
    ms_between_items = ss_between_items / (n - 1)
    ms_between_raters = ss_between_raters / (k - 1)
    ms_residual = ss_residual / ((n - 1) * (k - 1))

    # ICC(2,1) => two-way random, single measurement
    icc_value = (ms_between_items - ms_residual) / (
        ms_between_items + (k - 1) * ms_residual + (k / n) * (ms_between_raters - ms_residual)
    )
    return icc_value

def cronbach_alpha_two_raters(ra1, ra2):
    """
    Compute Cronbach's Alpha for two raters, given as two numeric arrays.
    For two raters, alpha = (2 * r_xy) / (1 + r_xy), where r_xy is Pearson correlation.
    """
    r_xy = np.corrcoef(ra1, ra2)[0, 1]
    alpha_val = (2.0 * r_xy) / (1.0 + r_xy)
    return alpha_val

def main():
    # 1) Load CSV
    df = pd.read_csv('dataframe.csv')

    #-------------------------------------------------------------------
    # 2) Overall metrics (use nth_trial==1 to get one AI score per response)
    #-------------------------------------------------------------------
    df_unique = df[df['nth_trial'] == 1].copy()
    y_true = df_unique['Actual_Score'].values
    y_pred = df_unique['o1_Score'].values

    exact_match_rate = np.mean(y_pred == y_true)
    within_one_rate  = np.mean(np.abs(y_pred - y_true) <= 1)

    kappa_unweighted = cohen_kappa_score(y_true, y_pred)
    kappa_quadratic  = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    icc_val   = intraclass_correlation(y_true, y_pred)
    alpha_val = cronbach_alpha_two_raters(y_true, y_pred)

    print("==== OVERALL METRICS ====")
    print(f"Exact Match Rate:       {exact_match_rate:.3f}")
    print(f"Within-One-Point Rate:  {within_one_rate:.3f}")
    print(f"Cohen's Kappa:          {kappa_unweighted:.3f}")
    print(f"Quadratic Weighted Kappa (QWK): {kappa_quadratic:.3f}")
    print(f"Pearson Correlation (r): {pearson_r:.3f}")
    print(f"Intraclass Correlation (ICC): {icc_val:.3f}")
    print(f"Cronbach's Alpha (2-rater): {alpha_val:.3f}\n")

    #-------------------------------------------------------------------
    # 3) Metrics by subgroup (e.g., Student_ID in {A, B, C})
    #-------------------------------------------------------------------
    print("==== METRICS BY STUDENT GROUP ====")
    groups = df_unique['Student_ID'].unique()
    for grp in groups:
        dfg = df_unique[df_unique['Student_ID'] == grp]
        y_true_g = dfg['Actual_Score'].values
        y_pred_g = dfg['o1_Score'].values

        exact_match_g = np.mean(y_pred_g == y_true_g)
        within_one_g  = np.mean(np.abs(y_pred_g - y_true_g) <= 1)
        kappa_g = cohen_kappa_score(y_true_g, y_pred_g)
        qwk_g   = cohen_kappa_score(y_true_g, y_pred_g, weights='quadratic')
        r_g     = np.corrcoef(y_true_g, y_pred_g)[0, 1]
        icc_g   = intraclass_correlation(y_true_g, y_pred_g)
        alpha_g = cronbach_alpha_two_raters(y_true_g, y_pred_g)

        print(f"Group {grp}:")
        print(f"  Exact Match Rate = {exact_match_g:.3f}")
        print(f"  Within-One-Point Rate = {within_one_g:.3f}")
        print(f"  Cohen's Kappa = {kappa_g:.3f}")
        print(f"  QWK = {qwk_g:.3f}")
        print(f"  Pearson r = {r_g:.3f}")
        print(f"  ICC = {icc_g:.3f}")
        print(f"  Cronbach's Alpha = {alpha_g:.3f}\n")

    #-------------------------------------------------------------------
    # 4) Inter-trial reliability (Cronbach's Alpha across 7 trials, etc.)
    #-------------------------------------------------------------------
    print("==== INTER-TRIAL RELIABILITY ====")
    # Pivot so that each row is a single response, each column is a trial's AI score
    pivot_df = df.pivot(
        index=['Year','Test_Type','Question','Subquestion','Student_ID'],
        columns='nth_trial',
        values='o1_Score'
    ).reset_index(drop=True)

    # pivot_df now has one row per unique response (should be 408 rows if that's the total),
    # and 7 columns (trials 1..7). We'll convert to a numpy array for Cronbach's Alpha.
    scores_matrix = pivot_df.to_numpy()  # shape = (N_responses, 7)

    # Calculate Cronbach's Alpha for 7 repeated measures
    K = scores_matrix.shape[1]  # number of trials (7)
    # Variance across responses for each trial
    var_per_trial = np.var(scores_matrix, axis=0, ddof=1)
    # Sum each row to get total score across all trials for that response
    total_scores = np.sum(scores_matrix, axis=1)
    var_total    = np.var(total_scores, ddof=1)

    # Cronbach's Alpha (classic formula for multi-item measure)
    alpha_7 = (K / (K - 1.0)) * (1.0 - np.sum(var_per_trial) / var_total)

    # As an example, also compute test-retest correlation between Trial 1 & Trial 7
    trial1 = scores_matrix[:, 0]
    trial7 = scores_matrix[:, 6]
    test_retest_r = np.corrcoef(trial1, trial7)[0, 1]

    print(f"Cronbach's Alpha across 7 trials: {alpha_7:.3f}")
    print(f"Test-Retest Correlation (Trial 1 vs. Trial 7): {test_retest_r:.3f}")

if __name__ == "__main__":
    main()

