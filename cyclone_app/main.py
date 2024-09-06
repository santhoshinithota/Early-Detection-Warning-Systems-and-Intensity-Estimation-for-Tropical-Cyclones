import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
import cv2
import backend.initialise as initialise
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", rain=1, x=0)

@app.route("/", methods=["POST"])
def satelite_model():

    imagefile = request.files["imagefile"]
    classification_result = "No cyclone detected"
    filename = secure_filename(imagefile.filename)
    image_path = os.path.join("static/uploads/", filename)
    imagefile.save(image_path)

    image = cv2.imread(image_path)
    image = np.array(image)
    totensor = transforms.ToTensor()
    image = totensor(image)
    resize = transforms.Resize(size=(250, 250))
    image = resize(image)
    image = image.unsqueeze(0)
    model = initialise.intialise()
    model.load_state_dict(
        torch.load("backend/satelite_model", map_location=torch.device("cpu"))
    )
    model.eval()
    predicted_intensity = torch.round(model(image)).item()
    print(predicted_intensity)

    if predicted_intensity >= 17 and predicted_intensity <= 27:
        classification_result = "Depression"
    elif predicted_intensity > 27 and predicted_intensity <= 33:
        classification_result = "Deep Depression"
    elif predicted_intensity > 33 and predicted_intensity <= 47:
        classification_result = "Cyclonic Storm"
    elif predicted_intensity > 47 and predicted_intensity <= 63:
        classification_result = "Severe Cyclonic Storm"
    elif predicted_intensity > 63 and predicted_intensity <= 89:
        classification_result = "Very Severe Cyclonic Storm"
    elif predicted_intensity > 89 and predicted_intensity <= 119:
        classification_result = "Extremely Severe Cyclonic Storm"
    elif predicted_intensity > 119:
        classification_result = "Super Cyclonic Storm"

    print("upload_image filename: ", filename)
    return render_template(
        "index.html",
        prediction=predicted_intensity,
        filename=filename,
        x=1,
        classification_result=classification_result,
    )


@app.route("/postmetadata", methods=["POST"])
def metadata_model():

    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    LWSW = request.form.get("LWSW")
    MWNE = request.form.get("MWNE")
    MWSE = request.form.get("MWSE")
    HWNW = request.form.get("HWNW")

    if latitude == "":
        latitude = 0
    if longitude == "":
        longitude = 0
    if LWSW == "":
        LWSW = 0
    if MWNE == "":
        MWNE = 0
    if MWSE == "":
        MWSE = 0
    if HWNW == "":
        HWNW = 0

    metadata_model = joblib.load("backend/metadata_model.pkl")
    custom_x_train = pd.DataFrame(
        columns=[
            "Latitude",
            "Longitude",
            "Low Wind SW",
            "Moderate Wind NE",
            "Moderate Wind SE",
            "High Wind NW",
        ]
    )
    custom_x_train["Latitude"] = [latitude]
    custom_x_train["Longitude"] = [longitude]
    custom_x_train["Low Wind SW"] = [LWSW]
    custom_x_train["Moderate Wind NE"] = [MWNE]
    custom_x_train["Moderate Wind SE"] = [MWSE]
    custom_x_train["High Wind NW"] = [HWNW]
    custom_y_pred = metadata_model.predict(custom_x_train)
    predicted_intensity = custom_y_pred[0]
    predicted_intensity = predicted_intensity.replace("TD", "Depression")
    predicted_intensity = predicted_intensity.replace("TS", "Cyclonic Storm")
    predicted_intensity = predicted_intensity.replace("HU", "Super Cyclonic Storm")

    if predicted_intensity == " Depression":
        isCyclone = True
    elif predicted_intensity == " Cyclonic Storm":
        isCyclone = True
    elif predicted_intensity == " Super Cyclonic Storm":
        isCyclone = True
    else:
        isCyclone = False
    return render_template(
        "index.html",
        prediction_metadata=predicted_intensity,
        longitude=longitude,
        latitude=latitude,
        LWSW=LWSW,
        MWNE=MWNE,
        MWSE=MWSE,
        HWNW=HWNW,
        x=2,
        isCyclone=isCyclone,
    )


if __name__ == "__main__":
    app.run(debug=True)
