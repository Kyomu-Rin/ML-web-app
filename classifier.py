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
        ans = request.form.get('sel')

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER) 
        os.mkdir(UPLOAD_FOLDER)

        table_result = ""
        count = 0
        for file in files:
            count += 1

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
            
            id = "<td>" + str(count) + "</td>"
            img = "<td><img src=" + filepath + "></td>"
            
            first_index = sorted_idx[0]
            first_ratio = y[first_index]
            first_label = labels[first_index]
            first_result = "<td><p>" + first_label + "</p><small>(" + str(round(first_ratio*100, 1)) + "%)</small></td>"

            second_index = sorted_idx[1]
            second_ratio = y[second_index]
            second_label = labels[second_index]
            second_result = "<td><p>" + second_label + "</p><small>(" + str(round(second_ratio*100, 1)) + "%)</small></td>"

            if first_label == ans:
                result = "<td>〇</td>"
            else:
                result = "<td>X</td>"

            table_result += ("<tr>" + id + img + first_result + second_result + result + "</tr>")

        return render_template("result.html", result=Markup(table_result), ans=Markup(ans))
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)