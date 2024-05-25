from ctr import *
from scipy import stats

def main():

    control_CTR = []
    treatment_CTR = []

    cctr, tctr = cal_ctr(0, 0.04)
    control_CTR.append(cctr)
    treatment_CTR.append(tctr)

    for i in range(4, 100, 2):
        print((i-2) / 100, (i+2)/100)
        cctr, tctr = cal_ctr((i-2) / 100, (i+2)/100)
        
        control_CTR.append(cctr)
        treatment_CTR.append(tctr)

    print(control_CTR)
    print(treatment_CTR)

    # Perform a paired t-test
    t_stat, p_value = stats.ttest_rel(control_CTR, treatment_CTR)

    print(f"Paired T-Test T-statistic: {t_stat}")
    print(f"Paired T-Test P-value: {p_value}")

    # Interpret results
    if p_value < 0.05:
        print("There is a statistically significant difference in CTR between the control and treatment groups.")
    else:
        print("There is no statistically significant difference in CTR between the control and treatment groups.")

if __name__ == "__main__":
    main()