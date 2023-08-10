from flask import Flask, render_template, redirect, url_for, request, flash
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__) #Initialise app
app.secret_key = 'your_secret_key_here'

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict.html', methods = ['POST', 'GET'])
def join():
    return render_template('join.html')

@app.route('/index.html')
def backtohome():
    return redirect(url_for('home'))

@app.route('/pred', methods = ['POST', 'GET'])
def predict():
    # Load Model
    yo = pickle.load(open('yo.pkl', 'rb'))
    # coloumn Name
    columnName = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    # Load Data
    result = request.form
    idx = int(request.form.get('idx'))
    limit_balance = int(request.form.get('limit_balance'))
    sex = int(request.form.get('sex'))
    education = int(request.form.get('education'))
    marriage = int(request.form.get('marriage'))
    age = int(request.form.get('age'))
    pay0 = int(request.form.get('pay0'))
    pay2 = int(request.form.get('pay2'))
    pay3 = int(request.form.get('pay3'))
    pay4 = int(request.form.get('pay4'))
    pay5 = int(request.form.get('pay5'))
    pay6 = int(request.form.get('pay6'))
    bill_amount1 = int(request.form.get('bill_amount1'))
    bill_amount2 = int(request.form.get('bill_amount2'))
    bill_amount3 = int(request.form.get('bill_amount3'))
    bill_amount4 = int(request.form.get('bill_amount4'))
    bill_amount5 = int(request.form.get('bill_amount5'))
    bill_amount6 = int(request.form.get('bill_amount6'))
    pay_amount1 = int(request.form.get('pay_amount1'))
    pay_amount2 = int(request.form.get('pay_amount2'))
    pay_amount3 = int(request.form.get('pay_amount3'))
    pay_amount4 = int(request.form.get('pay_amount4'))
    pay_amount5 = int(request.form.get('pay_amount5'))
    pay_amount6 = int(request.form.get('pay_amount6'))
    
    
    # Set values
    data = np.array(
        # O/P = 1
        [idx, limit_balance, sex, education, marriage, age, pay0, pay2, pay3, pay4, pay5, pay6, bill_amount1, bill_amount2, bill_amount3, bill_amount4, bill_amount5, bill_amount6, pay_amount1, pay_amount2, pay_amount3, pay_amount4, pay_amount5, pay_amount6]
    ).reshape(1, -1)
    
    x_arr = pd.DataFrame(data, columns=columnName)
    # Prediction
    poko1 = yo.predict(x_arr)
    a = (int(poko1[0]))
    flash('This is an alert!')
    
    # return render_template('op.html', data=a)
    return render_template('join.html', flash_message="True", data=a)
    # return redirect(url_for('join'), flash_message=True)



if __name__ == '__main__':
    app.run(debug=True)
    
