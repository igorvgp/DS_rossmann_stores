from flask import Flask, request, Response
import pickle
from rossmann.Rossmann import Rossmann
import pandas as pd


# loading model
model = pickle.load(open('C:/Users/igor/Documents/repos/ds_em_producao/Project/model/xgbtunned.pkl','rb'))

# Initiate API
app = Flask(__name__)

@app.route( '/rossmann/predict', methods=['POST'] )

def rossmann_predict():
    test_json = request.get_json()

    if test_json: # there is data
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame(test_json, index=[0]) # unique example
        else: 
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) # multiple example
        
        # Instantiate Rosmann class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )

        # Feature engineering
        df2 = pipeline.feature_engineering( df1 )

        # Data preparation
        df3 = pipeline.data_preparation ( df2 )

        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3)

        return df_response


    else:
        return Response ('{}', status = 200, mimetype = 'application/json')
    


if __name__ == '__main__':
    app.run( '10.0.0.175')
