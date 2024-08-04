from flask import Flask, render_template, request
import pickle
from car_data_prep import prepare_data
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


app = Flask(__name__)


def convert_to_float(value):
    try:
        return float(value.replace(',', ''))
    except:
        return np.nan


@app.route('/', methods=['POST','GET'])
def index():
    # Extracting all input fields from the form
    manufacturer    = request.form.get('manufacturer')
    year            = request.form.get('year')
    model           = request.form.get('model')
    hand            = request.form.get('hand')
    gear            = request.form.get('gear')
    capacity_engine = request.form.get('capacity_engine')
    engine_type     = request.form.get('engine_type')
    curr_ownership  = request.form.get('curr_ownership')
    area            = request.form.get('area')
    city            = request.form.get('city')
    price           = request.form.get('price')
    color           = request.form.get('color')
    km              = request.form.get('km')
    test            = request.form.get('test')
    
    # Convert inputs to the appropriate format for your model if needed
    car_info = pd.DataFrame({
        "manufactor":     [manufacturer],
        "Year":             [year],
        "model":            [model],
        "Hand":             [hand],
        "Gear":             [gear],
        "capacity_Engine":  [capacity_engine],
        "Engine_type":      [engine_type],
        "Curr_ownership":   [None],
        "Area":             [area],
        "City":             [city],
        "Price":            [None],
        "Color":            [color],
        "Km":               [km],
        "Test":             [None]
    })

    # car_info = pd.DataFrame({
    #     "manufactor":       ["ניסאן"],
    #     "Year":             [2018],
    #     "model":            ["ניסאן מיקרה"],
    #     "Hand":             [1],
    #     "Gear":             ["אוטומטית"],
    #     "capacity_Engine":  [1200],
    #     "Engine_type":      ["בנזין"],
    #     "Prev_ownership":   [None],
    #     "Curr_ownership":   [None],
    #     "Area":             ["מושבים בשרון"],
    #     "City":             ["אבן יהודה"],
    #     "Price":            [None],
    #     "Pic_num":          [None],
    #     "Cre_date":         [None],
    #     "Repub_date":       [None],
    #     "Description":      [None],
    #     "Color":            ["כחול בהיר"],
    #     "Km":               [10],
    #     "Test":             [None],
    #     "Supply_score":     [5]
    # })

# יונדאי,2015,i35,2,אוטומטית,1600,בנזין,פרטית,פרטית,רעננה - כפר סבא,רעננה,51000,2,11/7/23,11/7/23,['רכב שמור בקנאות\nמוכרת עקב קבלת רכב חברה'],כחול כהה מטאלי,144000,,

    # Preprocess inputs 
    df = pd.read_csv('dataset.csv')

    new_row = car_info.astype(car_info.dtypes.to_dict())
    A = pd.concat([new_row, df], ignore_index=True)

    A.to_csv('dataset.csv', index=False)

    df = pd.read_csv('dataset.csv')

    # car_info.reset_index(drop=True, inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # df = pd.concat([df, car_info], ignore_index=True)
    # df.loc[0] = car_info.iloc[0]

    df_prepared = prepare_data(df).drop(columns=['Price'])


    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # df.iloc[0, df.columns.get_loc('Km')] = float(100000)
    # prediction = df.head(1)["Km"]
    # prediction = model.predict(df_prepared.head(1)) 
    prediction = None if not manufacturer else model.predict(df_prepared.head(1)) 
    return render_template('index.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)

#for debug - if IP not showing chrome://net-internals/#sockets

