from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('G:\VS Code\projects\ML-Projects\Waiter Tip Prediction\model.pkl','rb'))

@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    bill = float(request.form['bill'])
    size = int(request.form['size'])
    gender = int(request.form['gender'])
    dinner_flag = int(request.form['df'])
    day = int(request.form['day'])
    
    input_features = np.array([[bill, size, gender, dinner_flag, day]])
    prediction = model.predict(input_features)
    
    return render_template('predict.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
