import pickle
from flask import Flask, request, jsonify, url_for, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

# create the app instance first
app = Flask(__name__)

# ===== debug wrapper for imports & model loads =====
print("app.py executing — starting import and model load...")

try:
    model = pickle.load(open('./models/Model.pkl', 'rb'))
    print("Loaded Model.pkl")
except Exception as e:
    print("ERROR loading Model.pkl:", repr(e))
    model = None

try:
    Brand_Encoder = pickle.load(open('./models/Brand_Encoder.pkl', 'rb'))
    print("Loaded Brand_Encoder.pkl")
except Exception as e:
    print("ERROR loading Brand_Encoder.pkl:", repr(e))
    Brand_Encoder = {}

try:
    Model_Encoder = pickle.load(open('./models/Model_Encoder.pkl', 'rb'))
    print("Loaded Model_Encoder.pkl")
except Exception as e:
    print("ERROR loading Model_Encoder.pkl:", repr(e))
    Model_Encoder = {}

try:
    OneHot_Encoder = pickle.load(open('./models/OneHot_Encoder.pkl', 'rb'))
    print("Loaded OneHot_Encoder.pkl")
except Exception as e:
    print("ERROR loading OneHot_Encoder.pkl:", repr(e))
    OneHot_Encoder = None

print("Import and model-load stage complete.")
# ===== form configuration for the frontend (read-only; does not touch the model) =====
def build_form_config():
    cfg = {'brand_models': {}, 'fuel_types': [], 'transmissions': [],
           'limits': {'year': None, 'engine': None, 'doors': None, 'owners': None, 'mileage': None}}
    try:
        cats = dict(zip(['Fuel', 'Transmission'], OneHot_Encoder.categories_))
        cfg['fuel_types'] = [str(v) for v in cats['Fuel']]
        cfg['transmissions'] = [str(v) for v in cats['Transmission']]
    except Exception as e:
        print("Form config: could not read OneHot_Encoder categories:", repr(e))
    try:
        df = pd.read_excel('raw_data.xlsx')
        for b, g in df.groupby('Brand'):
            cfg['brand_models'][str(b)] = sorted(str(m) for m in g['Model'].unique())
        rng = lambda c: {'min': float(df[c].min()), 'max': float(df[c].max())}
        cfg['limits'] = {'year': rng('Year'), 'engine': rng('EngineSize'), 'doors': rng('Doors'),
                         'owners': rng('OwnerCount'), 'mileage': rng('Mileage')}
        if not cfg['fuel_types']:
            cfg['fuel_types'] = sorted(str(v) for v in df['Fuel'].unique())
            cfg['transmissions'] = sorted(str(v) for v in df['Transmission'].unique())
    except Exception as e:
        print("Form config: could not read raw_data.xlsx:", repr(e))
    return cfg

FORM_CONFIG = build_form_config()
# ==================================================

# ==================================================
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html', form_config=FORM_CONFIG)


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Brand = request.form['Brand']
        Model = request.form['Model']
        Fuel = request.form['Fuel']
        Transmission = request.form['Transmission']
        Year = int(request.form['Year'])
        EngineSize = float(request.form['EngineSize'])
        Mileage = int(request.form['Mileage'])
        Doors = int(request.form['Doors'])
        OwnerCount = int(request.form['OwnerCount'])

        # Create a DataFrame for the input data
        input_df = pd.DataFrame({
            'Brand': [Brand],
            'Model': [Model],
            'Year': [Year],
            'EngineSize': [EngineSize],
            'Fuel': [Fuel],
            'Transmission': [Transmission],
            'Mileage': [Mileage],
            'Doors': [Doors],
            'OwnerCount': [OwnerCount]
        })

        # Encode Brand and Model using mean target encoding
        input_df['Encoded_Brand'] = input_df['Brand'].map(Brand_Encoder)
        input_df['Encoded_Model'] = input_df['Model'].map(Model_Encoder)
        input_df['Encoded_Brand'].fillna(input_df['Encoded_Brand'].mean(), inplace=True)
        input_df['Encoded_Model'].fillna(input_df['Encoded_Model'].mean(), inplace=True)
        input_df.drop(['Brand', 'Model'], axis=1, inplace=True)

        # One-hot encode Fuel and Transmission
        categorical_cols = ['Fuel', 'Transmission']
        encoded_array = OneHot_Encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=OneHot_Encoder.get_feature_names_out(categorical_cols))

        # Merge encoded columns with input data
        input_df_encoded = input_df.drop(columns=categorical_cols).reset_index(drop=True)
        input_data = pd.concat([input_df_encoded, encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        # If AJAX request, return JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            if output < 0:
                return jsonify({'prediction': None, 'message': "Sorry you cannot sell this car"})
            return jsonify({'prediction': output, 'message': f"Car is worth at: $ {output}"})

        # Regular form POST -> render template
        if output < 0:
            return render_template('index.html', form_config=FORM_CONFIG, prediction_text="Sorry you cannot sell this car")
        else:
            return render_template('index.html', form_config=FORM_CONFIG, prediction_text=f"Car is worth at: $ {output}")

    # fallback GET
    return render_template('index.html', form_config=FORM_CONFIG)
if __name__ == "__main__":
    try:
        print("Attempting to start Flask app...")
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print("FLASK START ERROR:", repr(e))
        raise