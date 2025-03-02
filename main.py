import google.generativeai as genai
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
import logging

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

classification_model = load_model('mobilenet_crop_disease_best.h5')
index_to_class = {
    0: "Corn__Common_Rust",
    1: "Corn__Gray_Leaf_Spot",
    2: "Corn__healthy",
    3: "Corn__Northern_Leaf_Blight",
    4 : "Potato___Early_blight",
    5 : "Potato___healthy",
    6 : "Potato___Late_blight",
    7 : "Rice__Healthy",
    8 : "Rice__Leaf_Blast",
    9 : "Rice__Neck_Blast",
    10: "Wheat_Brown_Rust",
    11: "Wheat_healthy",
    12: "Wheat_Yellow_Rust"
}

genai.configure(api_key = "YOUR_GEMINI_API_KEY")
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 512,
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = classification_model.predict(img_array, verbose=0)  # Set verbose=0 to silence TF output
    predicted_index = np.argmax(predictions, axis=1)[0]
    return index_to_class[predicted_index]

def recomendation(disease, soil_type, temperature, rainfall, humidity, n, p, k, ph):
    prompt = f"""
    I have the following details for a crop:
    - **Disease:** {disease}
    - **Soil Type:** {soil_type}
    - **Average Temperature (next 7 days):** {temperature}°C 🌡️
    - **Average Rainfall (next 7 days):** {rainfall} mm 🌧️
    - **Average Humidity (next 7 days):** {humidity}% 💧
    - **Soil Nutrient Values:** N = {n}, P = {p}, K = {k} 🧪
    - **Soil pH:** {ph}

    Based on these conditions, please provide:
    1. **Fertilizer/Manure Recommendations:** on the basis of data provide suggest the exact amount and name of fertilizers or manure to be used.
    2. **Outbreak Analysis:** Do the current soil and weather conditions support the disease? What are the chances of the disease spreading?
    3. **Remedies:** All possible remedies or treatments to manage or cure the disease.
    4. do not include /n /** these types of things in the output text.
    5. give concise and clear answers.
    6. do not include any unnecessary information.
    """

    response = model.generate_content(prompt)
    text_output = response.candidates[0].content.parts[0].text
    return text_output

def get_weather_forecast_averages(file_path='final_dataset.csv'):
    import sys
    original_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        current_date = df.index[-1]
        target_date = current_date + timedelta(days=7)
        steps = (target_date - current_date).days
        
        def fit_prophet_and_forecast(df, column, steps):
            series = df[column].interpolate().reset_index()
            series.columns = ['ds', 'y']
            model = Prophet(yearly_seasonality=True)
            model.fit(series)
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            forecast_values = forecast[['ds', 'yhat']].tail(steps)
            
            return forecast_values['yhat'].values
        
        def fit_prophet_for_rainfall(df, steps):
            rain_data = df.reset_index()[['Date', 'Rainfall_mm']]
            rain_data.columns = ['ds', 'y']
            rain_data['monsoon'] = rain_data['ds'].dt.month.isin([7, 8, 9])
            
            model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='monsoon', period=365.25, fourier_order=10, condition_name='monsoon')
            
            model.fit(rain_data)
            future = model.make_future_dataframe(periods=steps)
            future['monsoon'] = future['ds'].dt.month.isin([7, 8, 9])
            forecast = model.predict(future)
            forecast['yhat'] = np.clip(forecast['yhat'], 0, None)
            forecast['month'] = forecast['ds'].dt.month
            forecast.loc[forecast['month'].isin([7, 8, 9]), 'yhat'] = np.clip(forecast['yhat'], 0, 150)
            forecast.loc[forecast['month'].isin([12, 1, 2]), 'yhat'] = np.clip(forecast['yhat'], 0, 2)
            forecast_values = forecast[['yhat']].tail(steps)
            return forecast_values['yhat'].values
        
        temperature_forecast = fit_prophet_and_forecast(df, 'Temperature_C', steps)
        humidity_forecast = fit_prophet_and_forecast(df, 'Humidity_%', steps)
        rainfall_forecast = fit_prophet_for_rainfall(df, steps)
        avg_temperature = np.mean(temperature_forecast)
        avg_humidity = np.mean(humidity_forecast)
        avg_rainfall = np.mean(rainfall_forecast)
        
        averages = {
            'avg_temperature': round(avg_temperature, 2),
            'avg_humidity': round(avg_humidity, 2),
            'avg_rainfall': round(avg_rainfall, 2)
        }
        return averages
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def pipeline(file_path, soil_type, n, p, k, ph):

    disease = predict_image(file_path)
    weather_forecast = get_weather_forecast_averages()
    temperature = weather_forecast['avg_temperature']
    humidity = weather_forecast['avg_humidity']
    rainfall = weather_forecast['avg_rainfall']
    recommendations = recomendation(disease, soil_type, temperature, rainfall, humidity, n, p, k, ph)
    
    return disease , temperature , humidity , rainfall , recommendations


file_path = "Crop Diseases Dataset/Crop Diseases/Crop___Disease/Potato/Potato___Early_Blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG"
disease , temperature , humidity , rainfall , recomendations = pipeline(file_path, "Sandy", 20, 40, 60, 6.5)
print(recomendations)