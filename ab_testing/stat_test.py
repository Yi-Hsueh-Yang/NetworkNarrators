from scipy import stats

def compare_RMSE(RMSE_Old, RMSE_New, alpha=0.05):
    """
    Compare the RMSE values of two models using an independent t-test.

    Parameters:
    - RMSE_Old (list): List of RMSE values for the baseline model.
    - RMSE_New (list): List of RMSE values for the better model.
    - alpha (float): Significance level for the test (default is 0.05).

    Returns:
    - str: A string indicating whether there is a significant difference between the RMSE values of the two models.
    """
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(RMSE_Old, RMSE_New)

    # Print the results
    if p_value < alpha:
        return "Reject null hypothesis: There is a statistically significant difference."
    else:
        return "Fail to reject null hypothesis: There is no statistically significant difference."

# RMSE values for the two models from the online evaluation code, done for testing purposes 
RMSE_Old = [2.7543, 2.1300, 2.3469, 2.8567, 2.4904, 2.7670, 2.1088, 2.5451, 2.2249, 1.8571,
                 2.3188, 3.2909, 2.9329, 2.2812, 1.6257, 2.2143, 1.9513, 1.3315, 2.1261, 1.4822]
RMSE_New = [1.8414, 1.1091, 1.8933, 1.5169, 1.9062, 1.8878, 1.9403, 1.9367, 2.4482, 2.3014,
               1.6136, 1.8249, 2.3497, 2.0447, 1.4741, 1.7978, 1.3506, 1.5955, 1.0121, 1.3826]

print(compare_RMSE(RMSE_Old, RMSE_New))

