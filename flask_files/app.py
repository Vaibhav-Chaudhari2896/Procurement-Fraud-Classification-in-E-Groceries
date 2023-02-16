import numpy as np
from flask import Flask, request, render_template
import pickle
import json

app = Flask(__name__)
best_model = pickle.load(open('Best_Model.pickle', 'rb'))
with open('dummy_dict.json', 'r', encoding='utf-8') as f:
    dummy_dict = json.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = []
    for key, value in request.form.items():
        if key in ["Requesting_Auth_names", "Approving_manager_names", "Product_names", "Supplier_names", "City_names"]:
            int_features.extend(dummy_dict[key][value])
        elif key in ['Contract', 'PO_Quantity']:
            int_features.append(np.int(value))
        else:
            int_features.append(np.float(value))
    final_features = [np.array(int_features)]
    prediction = best_model.predict(final_features)

    if prediction == 0:
        final_result = 'The given transaction is Fraudulent'
    else:
        final_result = 'The given transaction is not Fraudulent'

    return str(final_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
