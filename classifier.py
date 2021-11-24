import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"

labels = ["ア","イ", "空欄", "ウ"]
n_class = len(labels)
img_size = 64
n_result = 2  # 上位2つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
        files = request.files.getlist('file')

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER) 
        os.mkdir(UPLOAD_FOLDER)

        result = ""
        for file in files:
            
            filename = secure_filename(file.filename)  # ファイル名を安全なものに
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像の読み込み
            image = Image.open(filepath)
            image = image.convert("RGB")
            image = image.resize((img_size, img_size))
            x = np.array(image, dtype=float)
            x = x.reshape(1, img_size, img_size, 3) / 255        

            # 予測
            model = load_model("./image_classifier.h5")
            y = model.predict(x)[0]
            sorted_idx = np.argsort(y)[::-1]  # 降順でソート
            
            

            for i in range(n_result):
                idx = sorted_idx[i]
                ratio = y[idx]
                label = labels[idx]
                result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)