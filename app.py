from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('gre.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    Chance_of_Admit = float(request.form.get('Chance_of_Admit'))
    TOEFL_Score = int(request.form.get('TOEFL_Score'))
    University_Rating = int(request.form.get('University_Rating'))
    CGPA = float(request.form.get('CGPA'))
    Research = int(request.form.get('Research'))

    input_query = np.array([[Chance_of_Admit,TOEFL_Score,University_Rating,CGPA,Research]])

    GRE_Score = model.predict(input_query)[0]

    return jsonify({'GRE_Score':float(GRE_Score)})

if __name__ == '__main__':
    app.run(debug=True)