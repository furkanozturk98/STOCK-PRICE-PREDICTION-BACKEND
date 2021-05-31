from flask import Flask
from flask_restful import Api, Resource
from flask import request

import math
import pandas_datareader as web
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import re
from datetime import datetime
from datetime import timedelta

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {
            "data":{
                "date": "2019-12-18",
                "prediction": "69.934998"
            }
        }

class prediction(Resource):
    def get(self):
        
        end_date = request.args.get('end_date')

        date = datetime.strptime(end_date, "%Y-%m-%d")
        date = date + timedelta(days=1)

        model = load_model('model.h5', compile = False)
        #Get the quote
        apple_quote = web.DataReader('AAPL', data_source='yahoo', start=request.args.get('start_date'), end=request.args.get('end_date'))
        #Create a new dataframe
        new_df = apple_quote.filter(['Close'])
        #Get the last 60 day closing price values and convert the dataframe to an array
        last_60_days = new_df[-60:].values

        scaler = MinMaxScaler(feature_range=(0,1))

        #Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.fit_transform(last_60_days)
        #Create an empty list
        X_test = []
        #Append the past 60 days
        X_test.append(last_60_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price[0][0])

        return {
            "data":{
                "date": str(date),
                "prediction": str(pred_price[0][0])
            }
        }

        """return {
            "start_date": request.args.get('start_date'),
            "end_date": request.args.get('end_date')
        } """

api.add_resource(HelloWorld, "/helloworld")
api.add_resource(prediction, "/predict")

if __name__ == "__main__":
    app.run(debug=True)

