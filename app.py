from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model_rf.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', heart_risk=None)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil 18 input dari form html
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Prediksi
        prediction = model.predict(final_features)

        if prediction[0] == 1.0:
            output = "Pasien Beresiko Terkena Penyakit Jantung"
        else:
            output = "Pasien Tidak Beresiko Terkena Penyakit Jantung"

        return render_template('index.html', heart_risk=output)
    except Exception as e:
        return render_template('index.html', heart_risk=f"Terjadi Kesalahan Input: {str(e)}")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

        