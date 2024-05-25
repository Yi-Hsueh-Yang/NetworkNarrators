import pandas as pd
from datetime import datetime
from fetch_ground_truth import *
from sys import argv

# enter `python ctr.py <rate>` to run the stats test, rate is (0, 1], indicating the amount of data we are using from the recommendation data

experiment_mapping = {'http://localhost:8084': 'treatment', 'http://localhost:8085': 'control'}

def load_recommendation_data():
    rec_df = pd.read_csv('filtered_recommendation.csv')
    rec_df['server_url'] = rec_df['server_url'].map(experiment_mapping)
    rec_df['recommendations'] = rec_df['recommendations'].str.replace("b'", "", regex=False)
    rec_df['recommendations'] = rec_df['recommendations'].str.split(',')
    return rec_df.explode('recommendations')

def load_click_data():
    click_df = pd.read_csv('click_df.csv')
    return click_df

def cal_ctr(start_percentage, end_percentage):

    click_df = get_data(start_percentage, end_percentage)
    rec_df = load_recommendation_data()
    # click_df = load_click_data()

    rec_df['userid'] = rec_df['userid'].astype(str)
    click_df['user_id'] = click_df['user_id'].astype(str)
    # print(rec_df.shape)
    # print(click_df.shape)
    rec_df.rename(columns={'recommendations': 'movie_id'}, inplace=True)
    rec_df.rename(columns={'userid': 'user_id'}, inplace=True)

    merged_data = pd.merge(rec_df, click_df, on=['user_id', 'movie_id'], how='left', indicator=True)

    # print(merged_data.shape)

    control_df = merged_data[merged_data['server_url'] == 'control']
    treatment_df = merged_data[merged_data['server_url'] == 'treatment']

    control_df.loc[:, 'clicked'] = (control_df['_merge'] == 'both').astype(int)  # 1 if clicked, 0 otherwise
    treatment_df.loc[:, 'clicked'] = (treatment_df['_merge'] == 'both').astype(int)  # 1 if clicked, 0 otherwise

    control_ctr = sum(control_df['clicked']) / len(control_df['clicked']) * 100
    treatment_ctr = sum(treatment_df['clicked']) / len(treatment_df['clicked']) * 100
    
    print("CTR for the control group:", control_ctr)
    print("CTR for the treatment group:", treatment_ctr)
    
    return control_ctr, treatment_ctr



