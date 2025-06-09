#solar_maintenance_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report,f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import cv2
import ast
import io
import os
import time
import joblib
import pvlib
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image

os.makedirs('static', exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)


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

    # Normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fault", "Fault"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Normalised Confusion Matrix – LSTM Fault Prediction")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Classification report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Fault", "Fault"]))

    # F1 score bar plot
    f1_scores = f1_score(y_test, y_pred, average=None)
    plt.bar(["No Fault", "Fault"], f1_scores, color=["skyblue", "steelblue"])
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("F1 Scores per Class")
    plt.tight_layout()
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
def create_app(model, scaler, yolo_model):
    """Create Flask API for predictions"""
    app = Flask(__name__)

    @app.route('/')
    def home (): return "Solar Maintenance Flask API is running!"
    
    @app.route('/predict', methods=['POST'])
    def predict():
    
        data = request.json['data']
        print ("Received data:", data)
        
        #Validate input length
        if len(data) != 3:
            return jsonify({'error': 'Expected 3 sensor inputs: [power,temp, irradiance]'}), 400
        
        #Transform and padding data
        scaled_data = scaler.transform([data])
        scaled_sequence = scaled_data[-24:]  # Use last 24 hours
        prediction = model.predict(scaled_sequence.reshape(1, 24, 3))

        return jsonify({
            'fault_probability': float(prediction[0][0]),
            'alert': 'Fault detected' if prediction > 0.5 else 'Normal'
        })
    
    
    @app.route('/detect-image', methods=['POST'])
    def detect_image():

        image = Image.open(request.files['image'].stream)
        results = yolo_model.predict(image)

        #Save annotated image
        annotated = results[0].plot()
        timestamp = int(time.time())
        filename = f'static/annotated_{timestamp}.jpg'
        cv2.imwrite(f'static/{filename}', annotated) #saved for results demo

        detections = [{'label':yolo_model.names[int(b.cls)],'confidence':float(b.conf)}
                      for r in results for b in r.boxes]
        return jsonify({
                        'detections': detections,
                        'alert': 'Defect detected' if any(d['label'] == 'defect' for d in detections) else 'No defect detected',
                        'image_url': f'/static/{filename}'
                    })

                 

    @app.route('/predict-combined', methods=['POST'])
    def predict_combined():
        try:
            image = Image.open(request.files['image'].stream)
            results = yolo_model.predict(image)
            
            #Generate and save annotated image with unique filename
            annotated = results[0].plot()
            timestamp = int(time.time())
            filename = f'annotated_{timestamp}.jpg'
            filepath = os.path.join(STATIC_DIR, filename)
            cv2.imwrite(filepath, annotated) #saved for results demo

                #Process YOLO detections
            detections = [{'label':yolo_model.names[int(b.cls)],'confidence':float(b.conf)}
                      for r in results for b in r.boxes]
            
            #Process sensor data from form-data
            sensor_data = ast.literal_eval(request.form['data'])
            scaled = scaler.transform([sensor_data])
            seq = np.tile(scaled,(24,1)).reshape(1, 24, 3)
            
            #Predict using LSTM
            pred = model.predict(seq)
            fault_prob = float(pred[0][0])
            lstm_alert = 'Fault detected' if fault_prob > 0.5 else 'Normal'
            yolo_alert = 'Defect detected' if any(d['label'] == 'defect' for d in detections) else 'No defect detected'

            return jsonify({
                'lstm_fault_probability': fault_prob,
                'lstm_alert': lstm_alert,
                'yolo_alert': yolo_alert,
                'yolo_detections': detections,
                'image_url': f'/static/{filename}?t={timestamp}'
            })
        
        except Exception as e:
            print("Error in /predict-combined:", e)
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

    #Load YOLOv8 model
    yolo_model = YOLO(r"C:\Users\eee_admin\ProjectDiss\Trial5\weights\best.pt")

    # 7. Deploy
    app = create_app(model, scaler, yolo_model)
    print("Flask API is running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)