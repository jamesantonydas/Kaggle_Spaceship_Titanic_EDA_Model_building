import pickle
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Please use /predict endpoint'

@app.route("/predict", methods=['POST'])
def do_prediction():
  
  data = request.form
  homeplanet = data.get('HomePlanet')
  cryosleep = data.get('CryoSleep')
  destination = data.get('Destination')
  age = data.get('Age')
  vip = data.get('VIP')
  room_service = data.get('RoomService')
  food_court = data.get('FoodCourt')
  shopping_mall = data.get('ShoppingMall')
  spa = data.get('Spa')
  vr_deck = data.get('VRDeck')
  deck = data.get('deck')
  side = data.get('side')

  df = pd.DataFrame([[homeplanet, cryosleep, destination, age, vip,
                      room_service, food_court, shopping_mall,
                      spa, vr_deck, deck, side]],
                    
                    columns=['HomePlanet', 'CryoSleep', 'Destination',
                             'Age', 'VIP', 'RoomService', 'FoodCourt',
                             'ShoppingMall', 'Spa', 'VRDeck', 'deck', 'side'])
  
  df[['HomePlanet', 'Destination', 'deck', 'side']] = df[['HomePlanet', 'Destination', 'deck', 'side']].astype('float')
  df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'CryoSleep']] = df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP', 'CryoSleep']].astype('float')

  model = pickle.load(open('xg_boost.pkl', 'rb'))
  y_predict = model.predict(df.values)
  result = "Transported" if y_predict[0] == 1 else "Not Transported"
  response = {'result' : result}
  return render_template('/result.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
