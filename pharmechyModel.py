from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import numpy as np
import os
from flask_cors import CORS  # Import CORS
app = Flask(__name__)
CORS(app)
# Ensure upload directory exists
UPLOAD_FOLDER = './excel-uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_sales_for_3_months(data):
    # Load and preprocess data
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')


    # Feature Engineering: Splitting the date
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Prepare for dynamic scaling
    medicines = data['Medicine'].unique()
    results = []

    for medicine in medicines:
        medicine_data = data[data['Medicine'] == medicine]
        medicine_data = pd.get_dummies(medicine_data, columns=['Medicine'])
        y = medicine_data['Amount Sold']
        X = medicine_data.drop(['Amount Sold', 'Profit', 'Total Revenue', 'Total Cost', 'Date'], axis=1)

        if len(medicine_data) > 5:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            mse_scores = -scores
            mse_rounded = round(np.mean(mse_scores), 2)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            future_date = datetime.now() + timedelta(days=90)
            future_data = X_test.copy()
            future_data['Year'] = future_date.year
            future_data['Month'] = future_date.month
            future_data['Day'] = future_date.day
            future_data = future_data[X.columns]

            future_sales = model.predict(future_data).mean()

            average_price = medicine_data['Price per Unit'].mean()
            average_cost = medicine_data['Cost per Unit'].mean()
            total_sales_volume = future_sales * 90
            total_revenue = total_sales_volume * average_price
            total_cost = total_sales_volume * average_cost
            profit = total_revenue - total_cost

            current_stock = medicine_data['Amount Sold'].sum() / len(medicine_data) * 90
            required_stock_increase = total_sales_volume - current_stock
            stocking_change_percentage = (required_stock_increase / current_stock) * 100 if current_stock else 0

            if stocking_change_percentage > 100:
                stocking_recommendation = f"Increase stock by {int(stocking_change_percentage)}% ({int(required_stock_increase)} units)"
            elif stocking_change_percentage > 50:
                stocking_recommendation = f"Increase stock moderately by {int(stocking_change_percentage)}% ({int(required_stock_increase)} units)"
            elif stocking_change_percentage > 0:
                stocking_recommendation = f"Increase stock slightly by {int(stocking_change_percentage)}% ({int(required_stock_increase)} units)"
            elif stocking_change_percentage < -50:
                stocking_recommendation = f"Decrease stock significantly by {int(-stocking_change_percentage)}% ({int(-required_stock_increase)} units)"
            else:
                stocking_recommendation = "Maintain current stock level"

            result = {
                'Medicine': medicine,
                'Cross-validated MSE': int(mse_rounded),
                'Predictions': {
                    'Sales Volume (units)': int(total_sales_volume),
                    'Revenue (₹)': int(total_revenue),
                    'Cost (₹)': int(total_cost),
                    'Profit (₹)': int(profit),
                    'Stocking Recommendation': stocking_recommendation
                }
            }
            results.append(result)
    return results

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                try:
                    df = pd.read_excel(filename)
                    predictions = predict_sales_for_3_months(df)
                    return jsonify(predictions)
                except Exception as e:
                    app.logger.error(f"Inner Error: {str(e)}")  # Log the inner error
                    return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"Outer Error: {str(e)}")  # Log the outer error
        return jsonify({'error': str(e)}), 500


    
if __name__ == '__main__':
    app.run()