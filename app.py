import datetime
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, DecimalField, DateField, SubmitField
from wtforms.validators import InputRequired, NumberRange
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'very_strong_password'

# FlaskForm for survey
class LoanSurveyForm(FlaskForm):
    gender = SelectField('Gender', choices=[('M', 'Male'), ('F', 'Female')], validators=[InputRequired()])
    own_car = SelectField('Do you own a car?', choices=[('Y', 'Yes'), ('N', 'No')], validators=[InputRequired()])
    own_estate = SelectField('Do you own real estate?', choices=[('Y', 'Yes'), ('N', 'No')], validators=[InputRequired()])
    mobile_flag = SelectField('Do you have a mobile?', choices=[('Y', 'Yes'), ('N', 'No')], validators=[InputRequired()])
    num_children = IntegerField('How many children do you have? (1, 2, etc.)', validators=[InputRequired(), NumberRange(min=0)])
    income_type = SelectField('Your income type', 
                              choices=[('Working', 'Working'), ('Commercial associate', 'Commercial associate'), 
                                       ('Pensioner', 'Pensioner'), ('State servant', 'State servant'), ('Student', 'Student')], 
                              validators=[InputRequired()])
    income_total = DecimalField('Income total ($ per year)', validators=[InputRequired()])
    education_type = SelectField('Education type', 
                                 choices=[('Secondary / secondary special', 'Secondary / secondary special'), 
                                          ('Higher education', 'Higher education'), 
                                          ('Incomplete higher', 'Incomplete higher'), 
                                          ('Lower secondary', 'Lower secondary'), 
                                          ('Academic degree', 'Academic degree')], 
                                 validators=[InputRequired()])
    family_status = SelectField('Family status', 
                                choices=[('Married', 'Married'), ('Single / not married', 'Single / not married'), 
                                         ('Civil marriage', 'Civil marriage'), ('Separated', 'Separated'), 
                                         ('Widow', 'Widow')], 
                                validators=[InputRequired()])
    housing_type = SelectField('Housing type', 
                               choices=[('House / apartment', 'House / apartment'), ('With parents', 'With parents'), 
                                        ('Municipal apartment', 'Municipal apartment'), ('Rented apartment', 'Rented apartment'), 
                                        ('Office apartment', 'Office apartment'), ('Co-op apartment', 'Co-op apartment')], 
                               validators=[InputRequired()])
    age = IntegerField('Your age', validators=[InputRequired(), NumberRange(min=18, max=100)])
    job_status = SelectField('Are you employed?', choices=[('Employed', 'Employed'), ('Unemployed', 'Unemployed')], validators=[InputRequired()])
    job_start_date = DateField('Date you entered your last job (Leave blank if Unemployed)', format='%Y-%m-%d', validators=[InputRequired()])
    mobile = SelectField('Do you have a mobile?', choices=[('1', 'Yes'), ('0', 'No')], validators=[InputRequired()])
    work_phone = StringField('Write your work phone (optional)', default='0')
    mobile_phone = StringField('Write your mobile (optional)', default='0')
    email = StringField('Email (optional)', default='0')
    occupation = StringField('Occupation (optional)', default='unknown')
    num_family_members = IntegerField('Number of family members', validators=[InputRequired(), NumberRange(min=0)])
    submit = SubmitField('Submit')

# Route for survey
@app.route('/', methods=['GET', 'POST'])
def survey():
    form = LoanSurveyForm()
    if form.validate_on_submit():
        # Collect data from form
        form_data = {
            'gender': form.gender.data,
            'own_car': form.own_car.data,
            'own_estate': form.own_estate.data,
            'mobile_flag': form.mobile_flag.data,
            'num_children': form.num_children.data,
            'income_type': form.income_type.data,
            'income_total': form.income_total.data,
            'education_type': form.education_type.data,
            'family_status': form.family_status.data,
            'housing_type': form.housing_type.data,
            'age': form.age.data,
            'job_start_date': form.job_start_date.data,
            'mobile': form.mobile.data,
            'work_phone': form.work_phone.data,
            'mobile_phone': form.mobile_phone.data,
            'email': form.email.data,
            'occupation': form.occupation.data,
            'num_family_members': form.num_family_members.data,
        }

        # Calculate employed_days or set it to 0 if unemployed
        if form.job_status.data == 'Employed':
            job_start_date = form.job_start_date.data
            employed_days = (datetime.datetime.now().date() - job_start_date).days
        else:
            employed_days = 0
        
        form_data['employed_days'] = employed_days
        
        # 1. Preprocess the form data
        prediction = predict(form_data)

        # 2. Save the form data and prediction to CSV
        save_to_csv(form_data, prediction)

        # 3. Return the result to the user
        return render_template('result.html', prediction=prediction)

    return render_template('survey.html', form=form)

# Prediction Function
def predict(data):
    # Convert form data to DataFrame
    input_data = pd.DataFrame([data])
    
    # Apply feature engineering
    from feature_engineering import preprocess_application_data
    processed_data = preprocess_application_data(input_data)
    
    # Load the model
    model = joblib.load('../models/catboost_model.pkl')
    
    # Make prediction (assuming the processed data matches model's required features)
    prediction = model.predict(processed_data)
    
    return "Approved" if prediction[0] == 1 else "Rejected"

# Function to save data and prediction to CSV
def save_to_csv(data, prediction):
    data['prediction'] = prediction
    df = pd.DataFrame([data])
    csv_file = os.path.join('data', 'loan_applications_with_predictions.csv')
    
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)

@app.route('/success')
def success():
    return "Survey submitted successfully!"

if __name__ == '__main__':
    app.run(debug=True)
