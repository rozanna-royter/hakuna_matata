from flask import Flask
from flask import request
import requests
import pandas as pd
import pickle
import os
import json

from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)


KELVIN_TO_C = 273.15
NUM_OF_DAYS = 7


@app.route('/')
def greetings():
    return 'Welcome to Hakuna Matata weather predictor. Use API /predict_7_days'


with open('weather_model.pickle', 'rb') as f:
    model = pickle.load(f)

# http://localhost:5000/predict_single?MedInc=7.33&HouseAge=28&AveRooms=4.55&AveBedrms=2.41&Population=299&AveOccup=2.66&Latitude=37.81&Longitude=-122.28
# http://localhost:5000/predict_single?own_t=12.85&wwo_t=13


@app.route('/predict_7_days')
def predict_7_days():
    own_temps = get_own()
    # print(own_temps)
    wwo_temps, dates = get_wwo()
    # print(wwo_temps)
    source_preds = []
    if len(own_temps) == NUM_OF_DAYS and len(wwo_temps) == NUM_OF_DAYS:
        for i in range(7):
            source_preds.append([round(own_temps[i], 2), float(wwo_temps[i])])
        # own_t = float(request.args.get('own_t'))
        # wwo_t = float(request.args.get('wwo_t'))
        # print(source_preds)

        poly_reg = PolynomialFeatures(degree=2)
        X_poly = poly_reg.fit_transform(source_preds)

        prediction = model.predict(X_poly)
        # prediction = model.predict([[19.93, 19.00], [20.09, 19.00], [19.3, 18.00], [15.06, 13.00], [16.78, 16.00], [19.31, 18.00], [14.73, 14.00]])

        date_predict = {}
        for i in range(len(dates)):
            date_predict[dates[i]] = prediction[i]
        print('Response:', date_predict)
        return date_predict
    else:
        return 'Error: Incorrect length of arrays. Check the API'


def get_own():
    url = 'https://api.openweathermap.org/data/2.5/onecall?lat=51.5074&lon=0.1278& exclude=minutely,hourly,daily&appid=d70a2538813f2708ceb4e3e12a9cd4a9'
    r = requests.get(url)
    own_json = r.json()
    daily = own_json['daily'][1:]
    daily_temps = [x['temp']['day']-KELVIN_TO_C for x in daily]
    # print(f'OWN answer: {r.json()}')
    return daily_temps


def get_wwo():
    url = 'http://api.worldweatheronline.com/premium/v1/weather.ashx?key=2120c3bc36124169b2573330202508&q=London&format=json&num_of_days=8'
    r = requests.get(url)
    if r.status_code == 200:
        wwo_json = r.json()
    else:
        with open('wwo_json') as json_file:
            data = json.load(json_file)
        wwo_json = data
    daily = wwo_json['data']['weather'][1:]
    daily_temps = [x['hourly'][4]['tempC'] for x in daily]
    dates = [x['date'] for x in daily]
    # print(f'WWO answer: {r.json()}')
    return daily_temps, dates


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
