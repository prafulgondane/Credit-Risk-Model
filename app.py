from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
# https://www.tutorialspoint.com/flask
import flask
import pickle
app = Flask(__name__)



# load model
model = joblib.load('model_l.pkl')
#model = joblib.load('model_j.pkl')#
#model = pickle.load(open("model_p.pkl", 'rb')) 
data = pd.read_csv('LGBM_TEST_DATA.csv')

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/get_loan_details', methods=['POST'])
def get_loan_details():
    print('Inside get_loan_details')
    to_predict_list = request.form.to_dict()
    id = to_predict_list['loan_id']
    d = data.loc[data['SK_ID_CURR'] == int(id)]
    print('Data :', d)
    gender = "M" if int(d['CODE_GENDER_L'].values[0]) > 0 else "F"
    loan_id = str(d['SK_ID_CURR'].values[0])
    income = str(d['AMT_INCOME_TOTAL'].values[0])
    loan_amt = str(d['AMT_CREDIT'].values[0])
    print('Data :', loan_id, gender,income , loan_amt)
    #return jsonify({'LOAN_ID' : d['SK_ID_CURR'].values[0], 'GENDER': gender , 'INCOME': d['AMT_INCOME_TOTAL'].values[0], 'LOAN_AMT' : d['AMT_CREDIT'].values[0]})
    return jsonify({'loan_id': loan_id, 'gender' : gender, 'income' : income, 'loan_amt' : loan_amt})

@app.route('/predict', methods=['POST'])
def predict():
    print('Inside predict')
    #lr = joblib.load('model.pkl')
    #lr = joblib.load('model_j.pkl')
    #lr = joblib.load('model_l.pkl')
    
    #lr = pickle.load(open('model_p.pkl', 'rb')) 
    #lr = joblib.load('model_j.pkl')
    print(request)
    print(request.form.to_dict())
    to_predict_list = request.form.to_dict()
    
    
    #to_predict_list = list(to_predict_list.values())
    #to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
    #print(to_predict_list,to_predict_list.shape)
    #prediction = lr.predict(to_predict_list)
    
    id = to_predict_list['loan_id']
    print('ID :', id)
    
    print('Data Head:', data.head(2))
    
    d = data.loc[data['SK_ID_CURR'] == int(id)].values[0][0:-1]
    
    #print('Data :', d.values())
    #print('lst :', lst)
    
    #to_predict_list = list(to_predict_list.values())
    #to_predict_list = list(d.values())
    to_predict_list = np.array(list(map(float, d))).reshape(1, -1)
    #print(to_predict_list[0:3], to_predict_list.shape)
    
    #print(to_predict_list, to_predict_list.shape)
    prediction = model.predict_proba(to_predict_list)[:, 1]
    print("prediction : ", prediction)
    return jsonify({'prediction': list(prediction)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
