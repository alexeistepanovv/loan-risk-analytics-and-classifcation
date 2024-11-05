
import pandas as pd
import numpy as np
from main import create_category_buckets, calculate_woe_iv, convert_to_dummy

def preprocess_application_data(application):
    # Handling missing values and transformations
    application['occupation'].fillna('unknown', inplace=True)
    application['gender'] = application['gender'].replace(['M','F'],[0,1])
    application['own_car'] = application['own_car'].replace(['N','Y'],[0,1])
    application['own_estate'] = application['own_estate'].replace(['N','Y'],[0,1])
    
    # Label handling
    labeled_application = application[application['label'] != 'new']
    labeled_application['label'] = pd.to_numeric(labeled_application['label'], errors='coerce').astype('int8')
    
    # Feature transformations
    application['income_total'] = application['income_total'] / 10000
    application['age_y'] = -(application['age_days']) // 365
    application['employed_y'] = -(application['employed_days']) // 365
    
    # Fill NaNs based on specific rules
    application.loc[application['income_type'] == 'Pensioner', 'employed_y'] = np.nan
    application.loc[(application['employed_y'] < 0) & (application['income_type'] != 'Pensioner'), 'employed_y'] = np.nan
    application['employed_y'].fillna(application['employed_y'].mean(), inplace=True)
    
    # Replace values based on number of children
    application.loc[application['num_children'] >= 2, 'num_children'] = '2More'
    application.loc[application['num_family_members'] >= 4, 'num_family_members'] = '4More'
    
    # Income and age bucketing
    application = create_category_buckets(application, 'income_total', 5, ["lowest", "low", "medium", "high", "highest"])
    application = create_category_buckets(application, 'age_y', 5, ["youngest", "young", "middle", "old", "oldest"])
    application = create_category_buckets(application, 'employed_y', 5, ["lowest", "low", "medium", "high", "highest"])
    
    # Occupation grouping
    application.loc[application['occupation'].isin(['Cleaning staff', 'Cooking staff', 'Drivers', 'Laborers', 'Low-skill Laborers', 'Security staff', 'Waiters/barmen staff']), 'occupation'] = 'Labor Work'
    application.loc[application['occupation'].isin(['Accountants', 'Core staff', 'HR staff', 'Medicine staff', 'Private service staff', 'Realty agents', 'Sales staff', 'Secretaries']), 'occupation'] = 'Office Work'
    application.loc[application['occupation'].isin(['Managers', 'High skill tech staff', 'IT staff']), 'occupation'] = 'High Tech Work'
    # List of categorical variables to convert to dummies
    dummy_vars = ['num_children', 'income_type', 'education_type', 'family_status', 
                'housing_type', 'occupation', 'num_family_members', 
                'gp_income_total', 'gp_age_y', 'gp_employed_y']

    # Apply the convert_to_dummy function for each categorical variable
    for var in dummy_vars:
        application = convert_to_dummy(application, var)
    
    application = application[['gender', 'own_car', 'own_estate', 'mobile_flag', 'work_phone',
    'num_children_1', 'num_children_2More',
    'income_type_Commercial associate', 'income_type_State servant',
    'education_type_Academic degree', 'education_type_Higher education',
    'education_type_Incomplete higher', 'education_type_Lower secondary',
    'family_status_Civil marriage', 'family_status_Separated',
    'family_status_Single / not married', 'family_status_Widow',
    'housing_type_Co-op apartment', 'housing_type_Municipal apartment',
    'housing_type_Office apartment', 'housing_type_Rented apartment',
    'housing_type_With parents', 'occupation_High Tech Work',
    'occupation_Labor Work', 'occupation_Office Work',
    'num_family_members_1.0', 'num_family_members_3.0',
    'num_family_members_4More', 'gp_income_total_low',
    'gp_income_total_medium', 'gp_income_total_high',
    'gp_income_total_highest', 'gp_age_y_young', 'gp_age_y_middle',
    'gp_age_y_old', 'gp_age_y_oldest', 'gp_employed_y_low',
    'gp_employed_y_medium', 'gp_employed_y_high', 'gp_employed_y_highest']]
    
    return application