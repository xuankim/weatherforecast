import pandas
import numpy as np
from joblib import load
import json
import random

with open('C:\Users\Admin\Desktop\WeatherForecast\json\dataWeather_Condition.json',"r",encoding="utf-8") as f:
    WeatherConditions = json.load(f)
# f = open('dataWeather_Condition.json',encoding="utf-8")
# text = f.read()
# print(len(text))
# WeatherConditions = json.load(f, strict=False)

clf = load('C:\Users\Admin\Desktop\WeatherForecast\model\DecisionTreeModel.joblib')

def PredictWeather(nhietdo,doam,mucgio,may,luongmua,luongtuyet):
  predicted = clf.predict([[nhietdo,doam,mucgio,may,luongmua,luongtuyet]])
  x = WeatherConditions[str(predicted[0])]
  x['notice'] = random.choice(WeatherConditions['notice'][x['name']])
  return x
