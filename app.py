from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('svc_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        form = request.form
        name = form['name']
        age = int(form['age'])
        sex = int(form['sex'])
        cp = int(form['cp'])
        trestbps = float(form['trestbps'])
        chol = float(form['chol'])
        fbs = int(form['fbs'])
        restecg = int(form['restecg'])
        thalach = float(form['thalach'])
        exang = int(form['exang'])
        oldpeak = float(form['oldpeak'])
        slope = int(form['slope'])

        data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope]]

        prediction = model.predict(data)

        if prediction == 2:
            message = 'You are likely to have heart disease.'
        else:
            message = 'You are not likely to have heart disease.'

        return render_template('result.html', name=name, result=message)

    except Exception as e:
        return render_template('result.html', name="Error", result=f"Something went wrong: {e}")
if __name__ == '__main__':
    app.run(debug=True)
