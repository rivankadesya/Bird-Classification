import base64
import numpy as np
from io import BytesIO
from model import Model
from preprocessing import Preprocessing
from inference import Inference
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

class Main:
    def __init__(self, img_input):
        self.img_input = img_input
        self.inference()
        self.postpreprocessing()

    def get_model(self):
        model = Model()
        model = model.get_model
        return model

    def prepare_input(self):
        prep = Preprocessing(img_name=self.img_input)
        prep.load_image()
        prep = prep.get_image_
        return prep

    def inference(self):
        model = self.get_model()
        prep = self.prepare_input()
        print(model)
        model = Inference(model, prep)
        self.result = model.infer()

    def postpreprocessing(self):
        dict_map = {
            0:"Elang Bido",
            1:"Jalak Bali",
            2:"Barn Owl",
            3:"Kasuari",
            4:"Enggano Myna",
            5:"Elang Ikan Kepala Abu",
            6:"Gelatik Jawa",
            7:"Maleo",
            8:"Oriental Bay Owl",
            9:"Merak"
        }
        label = dict_map[self.result.tolist()[0]]
        self._label = label

    @property
    def get_results(self):
        return self._label



    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.form.get('source') == 'file':
        file = request.files['image']
        image_data = file.read()
    else:
        image_data = base64.b64decode(request.form.get('image'))
    
    buf = BytesIO(image_data)
    data = buf.getvalue()
    encoded = base64.b64encode(data)
    encoded_string = encoded.decode('utf-8')
    
    model_img = Main(img_input=encoded_string)
    print(model_img.get_results)
    print(model_img.get_results)
    return jsonify({'predicted': model_img.get_results})

if __name__ == '__main__':
    app.run(debug=True)
    
    
