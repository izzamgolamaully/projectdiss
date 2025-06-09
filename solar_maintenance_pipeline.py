#solar_maintenance_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import pvlib
from flask import Flask, request, jsonify

# -------------------------
# 1. Data Synthesis & Preprocessing
# -------------------------
def generate_synthetic_data():
    """Generate synthetic solar data with injected faults"""
    # Nottingham location setup
    location = pvlib.location.Location(
        latitude=52.9548, longitude=-1.1581, tz='Europe/London', 
        altitude=61, name='Nottingham'
    )

    # Get TMY data
    tmy_data, _, _, _ = pvlib.iotools.get_pvgis_tmy(
        latitude=location.latitude,
        longitude=location.longitude,
        map_variables=True
    )
    tmy_data = tmy_data.rename(columns={
        'Gb(n)': 'dni', 'G(h)': 'ghi', 'Gd(h)': 'dhi'
    })

    # Simulate PV system
    system = pvlib.pvsystem.PVSystem(
        surface_tilt=30,
        surface_azimuth=180,
        module_parameters={'pdc0': 300, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 300},
        racking_model='open_rack',
        module_type='glass_polymer'
    )

    # Model chain setup
    mc = pvlib.modelchain.ModelChain(
        system=system,
        location=location,
        aoi_model='physical',
        spectral_model='no_loss',
        temperature_model='sapm'
    )
    mc.run_model(tmy_data)

    # Create DataFrame with synthetic faults
    df = pd.DataFrame({
        'power': mc.results.ac,
        'temp': mc.results.cell_temperature,
        'irradiance': mc.results.effective_irradiance
    })

    # Inject synthetic faults (20% of data)
    fault_indices = df.sample(frac=0.2).index
    df['fault'] = 0
    df.loc[fault_indices, 'fault'] = 1
    df.loc[fault_indices, 'power'] *= np.random.uniform(0.5, 0.8, len(fault_indices))
    
    return df

def preprocess_data(df):
    """Clean and normalise data"""
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    # Assuming the last column is the target
    features = solar_df.iloc[:, :-1].values
    target = solar_df.iloc[:, -1].values

    # Normalise features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['power', 'temp', 'irradiance']])
    
    # Save scaler for deployment
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return scaled_data, df['fault'].values, scaler

# -------------------------
# 2. Anomaly Detection & Auto-Labeling
# -------------------------
def auto_label_features(X):
    """Detect anomalies using Isolation Forest"""
    clf = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    anomalies = clf.fit_predict(X)
    return np.where(anomalies == -1, 1, 0)  # Convert to binary labels

# -------------------------
# 3. LSTM Model Training
# -------------------------
def create_sequences(data, labels, window_size=24):
    """Create time-series sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Construct LSTM architecture"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------
# 4. Model Evaluation
# -------------------------
def evaluate_model(model, X_test, y_test, history):
    """Generate performance metrics and plots"""
    # Generate predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Classification report
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History')
    plt.legend()
    plt.show()

    #Plot Loss curves
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Plotting prediction vs actual faults
    plt.plot(y_test, label='Actual Faults')    
    plt.plot(y_pred, label='Predicted Faults')
    plt.title('Actual vs Predicted Faults')
    plt.xlabel('Sample Index')
    plt.ylabel('Fault Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Error Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
    

# -------------------------
# 5. Model Deployment (Flask API)
# -------------------------
def create_app(model, scaler):
    """Create Flask API for predictions"""
    app = Flask(__name__)

    @app.route('/')
    def home(): return "Solar Maintenance Flask API is running!"

    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json['data']
            print("Received data:", data)

            # Validate input
            if len(data) != 3:
                return jsonify({'error': 'Expected 3 sensor values: [power, temp, irradiance]'}), 400

        # Scale the data and simulate a 24-hour window
            scaled_data = scaler.transform([data])
            sequence = np.tile(scaled_data, (24, 1))  # repeat the row 24 times
            sequence = sequence.reshape(1, 24, 3)

        # Make prediction
            prediction = model.predict(sequence)
            fault_prob = float(prediction[0][0])

            return jsonify({
            'fault_probability': fault_prob,
            'alert': 'Fault detected' if fault_prob > 0.5 else 'Normal'
            })

        except Exception as e:
            print("Error in /predict:", e)
        return jsonify({'error': str(e)}), 500

    
    return app

# -------------------------
# Main Execution Pipeline
# -------------------------
if __name__ == "__main__":
    # 1. Generate and preprocess data
    solar_df = generate_synthetic_data()
    X, y, scaler = preprocess_data(solar_df)
    
    # 2. Auto-labeling
    auto_labels = auto_label_features(X)
    
    # 3. Prepare LSTM sequences
    window_size = 24
    X_seq, y_seq = create_sequences(X, auto_labels, window_size)
    
    # Train/test split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # 4. Build and train model
    model = build_lstm_model((window_size, 3))
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # 5. Evaluate
    evaluate_model(model, X_test, y_test, history)
    
    # 6. Save model and scaler
    model.save('models/solar_lstm.h5')
    
    # 7. Deploy
    app = create_app(model, scaler)
    app.run(host='0.0.0.0', port=5000)