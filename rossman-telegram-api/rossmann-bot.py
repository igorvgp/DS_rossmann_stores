import pandas as pd
import requests
import json
from flask import Flask, request, Response

#constants
TOKEN = '5632821313:AAFnIJChckEaJ01wbC0XH9EuimgeVJQglZQ'

#   Info about the bot
#https://api.telegram.org/bot5632821313:AAFnIJChckEaJ01wbC0XH9EuimgeVJQglZQ/getMe

# Get Updates
#https://api.telegram.org/bot5632821313:AAFnIJChckEaJ01wbC0XH9EuimgeVJQglZQ/getUpdates

# Webhook
#https://api.telegram.org/bot5632821313:AAFnIJChckEaJ01wbC0XH9EuimgeVJQglZQ/setWebhook?url=https://9b5f-170-239-75-95.sa.ngrok.io/

# Send message
#https://api.telegram.org/bot5632821313:AAFnIJChckEaJ01wbC0XH9EuimgeVJQglZQ/sendMessage?chat_id=373642690&text=Hi Igor, I'm doing great, tks!

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)
    r = requests.post(url, json={'text':text})
    print('Status Code {}'.format(r.status_code))

    return None

def load_dataset(store_id):
    # Loading dataset
    test_data = pd.read_csv('test.csv')
    df_store_raw = pd.read_csv('store.csv', low_memory = False)

    # merge test dataset + store
    df_test = pd.merge( test_data, df_store_raw, how = 'left', on = 'Store' )

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        # drop ID
        df_test = df_test.drop('Id', axis = 1 )

        # convert Dataframe to json
        data = json.dumps( df_test.to_dict( orient='records' ) )
    else:
        data = 'error'

    return data

def predict(data):
    # API Call
    url =  'https://rossmann-model-new-app.herokuapp.com/rossmann/predict'  
    header = {'Content-type': 'application/json' }
    data = data
    r = requests.post( url, data=data, headers=header )
    print( 'Status Code {}'.format( r.status_code ) )
    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())
    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'

    return chat_id, store_id

# API initialize
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method =='POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            # Loading data
            data = load_dataset(store_id)
            if data != 'error':
                # prediction
                d1 = predict(data)
                # calculation
                d2 = d1[['store','prediction']].groupby('store').sum().reset_index()
                # send message
                msg = "Store number {} will sell ${:,.2f} in the next 6 weeks".format(d2['store'].values[0], d2['prediction'].values[0])
                send_message(chat_id, msg)
                return Response('Ok', status = 200)
            else:
                send_message(chat_id, "Store Not Available")
                return Response('Ok', status = 200)            
        
        else:
            send_message(chat_id, "Wrong Store ID")
            return Response('Ok', status = 200)
    else:
        return '<h1> Rossmann Telegram BOT </h1>' 

if __name__=='__main__':
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0', port=port)