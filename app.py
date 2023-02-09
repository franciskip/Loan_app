# Importing essential libraries
from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

# Load the Extra Tree CLassifier model
filename = 'Pickle_Logistic_Model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        
        Income = float(request.form['Income'])
        Savings = int(request.form['Savings'])
        Years_in_job = int(request.form['Years_in_job'])
        Home_ownership = request.form.get('Home_ownership')
        Open_accounts = int(request.form["Open_accounts"])
        Credit_cards = int(request.form["Credit_cards"])
        Overdraft = int(request.form['Overdraft'])
        Student_Loan = request.form.get("Student_Loan")
        Non_perfoming_Accs = int(request.form['Non_perfoming_Accs'])
        Current_In_Arrears = int(request.form["Current_In_Arrears"])
        Current_balance_Amt = int(request.form["Current_balance_Amt"])
        Past_due_Amt = int(request.form["Past_due_Amt"])
        
        
        data = np.array([[Income, Savings, Years_in_job, Home_ownership, Open_accounts, 
        Credit_cards, Overdraft, Student_Loan, Non_perfoming_Accs, Current_In_Arrears, 
        Current_balance_Amt, Past_due_Amt]])
 
        data = data.astype('float')
        my_prediction = model.predict(data)
         
        return render_template('results.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

