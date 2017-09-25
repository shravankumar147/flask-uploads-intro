from flask import Flask, render_template, request
from flask_uploads import UploadSet, IMAGES, configure_uploads

app = Flask(__name__)

photos = UploadSet('photos', [IMAGES, 'pgm'])
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage.color import rgb2gray
from skimage import exposure, feature, transform
from flask import jsonify, json

svc = joblib.load('./clf/clf_svc_hog.pkl')

def feature_extract(image):
    gray_img = rgb2gray(image)
    (image_feat, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8,8),
    cells_per_block=(2,2), transform_sqrt=True, visualise=True)
    image_feat = image_feat.reshape(1, -1)
    return image_feat

def predict_face(im_path):
    image =plt.imread(im_path)
    image_feat = feature_extract(image)
    pred_label = svc.predict(image_feat)[0]
    result = json.dumps({'results':pred_label},indent=4)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        f = os.path.join(app.config['UPLOADED_PHOTOS_DEST'],filename)
        return predict_face(f)
#         return predict_face('static/img'+'/'+filename)
    return render_template('upload.html')


if __name__ == '__main__':
	app.run(debug=True)