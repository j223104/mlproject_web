from flask import Flask, render_template, request,jsonify
import requests
import pandas as pd
import matplotlib.dates as mdates
from mplfinance import original_flavor as mpl
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import numpy as np
data = "오류"
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/mlbitcoin1", methods=["GET"])
def get_img():
    global pred
    ##########################################이미지만들기##################################
    COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades',
               'tb_base_av', 'tb_quote_av', 'ignore']
    URL = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '5m',
        'limit': 20, }
    result = requests.get(URL, params=params)
    js = result.json()
    df = pd.DataFrame(js, columns=COLUMNS)
    df.set_index('Open_time')
    df['Open_time'] = df['Open_time'].map(mdates.date2num)
    df = df.astype(float)
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(50 / 96,
                              50 / 96), dpi=96)
    ax1 = fig.add_subplot(1, 1, 1)
    mpl.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'],
                          width=1, colorup='#77d879', colordown='#db3f3f')
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis('off')
    ########################## fig 에 이미지 생김 그걸 png파일로 저장 ##################################

    fig.savefig('static/images/img_file.png', pad_inches=0, transparent=False)
    from PIL import Image
    img = Image.open('static/images/img_file.png')
    img = img.convert('RGB')
    img.save('static/images/img_file.png')

    ####################################html파일로 변환해서 보내줄 목적###############################
    # tmp_file = BytesIO()
    # fig.savefig(tmp_file, format='png')
    # encoded = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
    # data = f'data:image/png;base64,{encoded}'  ##
    ######################################################################################################
    model = tf.keras.models.load_model('model.h5')
    my_image = load_img('static/images/img_file.png', target_size=(224, 224))
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)
    prediction = model.predict(my_image)
    y_pred = np.argmax(prediction, axis=-1)

    if y_pred == [0]:
        data = "0"

    elif y_pred == [1]:
        data = "1"

    elif y_pred == [2]:
        data = "2"
    else:
        print(y_pred)
    print(data)

    return data


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)