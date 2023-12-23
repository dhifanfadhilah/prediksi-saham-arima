import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template, request
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

def train_arima(data):
    train_data, test_data = data[0:int(len(data)*0.5)], data[int(len(data)*0.5):]
    train_arima = train_data['open']
    test_arima = test_data['open']

    history = [x for x in train_arima]
    y = test_arima
    # make first prediction
    predictions = list()
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])

    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs) 

    last_close_price = data['close'].iloc[-1]
    future_price = predictions[-1]
    price_diff = future_price - last_close_price

    if price_diff > 0:
        recommendation = "Rekomendasi: Beli saham. Prediksi kenaikan sebesar {:.2f}".format(abs(price_diff))
    else:
        recommendation = "Rekomendasi: Jual saham. Prediksi penurunan sebesar {:.2f}".format(abs(price_diff))

    mse = str(mean_squared_error(y, predictions))
    mae = str(mean_absolute_error(y, predictions))
    rmse = str(math.sqrt(mean_squared_error(y, predictions)))

    plt.figure(figsize=(16,8))
    plt.plot(data.index[-600:], data['open'].tail(600), color='green', label = 'Latihan Harga Saham')
    plt.plot(test_data.index, y, color = 'red', label = 'Harga Saham Sebenarnya')
    plt.plot(test_data.index, predictions, color = 'blue', label = 'Prediksi Harga Saham')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Market')
    plt.legend()
    plt.grid(True)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url, mse, mae, rmse, recommendation

# Define the web application routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file, index_col="timestamp", parse_dates=True)
            
            if 'close' in data.columns:
                
                plot_url, mse, mae, rmse, recommendation = train_arima(data)

                return render_template('index.html', plot_url=plot_url, mse=mse, mae=mae, rmse=rmse, recommendation=recommendation)
    
    return render_template('index.html', plot_url=None, mse=None, mae=None, rmse=None, recommendation=None)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)