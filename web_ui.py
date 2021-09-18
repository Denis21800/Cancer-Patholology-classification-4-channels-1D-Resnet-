from flask import render_template, request, make_response, jsonify, flash, redirect, Flask, url_for
from werkzeug.utils import secure_filename
from pipeline import PipelineProcessor
from utils import ModelOutData
import pandas as pd
import plotly.express as px
import os

UPLOAD_FOLDER = '/home/dp/TMP/'
ALLOWED_EXTENSIONS = {'mgf'}

app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
config_path = './config/config.json'

global current_file_path
global current_results


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_mgf():
    global current_file_path
    if request.method == "POST":
        file = request.files["file"]
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(current_file_path)
            res = make_response(jsonify({"message": f"File {filename} uploaded"}), 200)
        else:
            res = make_response(jsonify({"message": " This file type is not allowed"}), 300)

        return res
    return render_template("upload_mgf.html")


@app.route("/predict_file/", methods=["GET", "POST"])
def predict_file():
    global current_results
    predictor = Predictor(config_path, UPLOAD_FOLDER)
    data = predictor.predict_file(current_file_path)
    current_results = data.to_html(classes='female')
    return render_template('predict_mgf.html', tables=[current_results], titles=[''])


@app.route("/show_data/", methods=["GET", "POST"])
def show_data():
    global current_results
    if request.method == "POST":
        selected_item = request.form['comp_select']
        predictor = Predictor(config_path, UPLOAD_FOLDER)
        if selected_item == '0':
            data = predictor.load_original(current_file_path)
        elif selected_item == '1':
            data = predictor.load_cleaned(current_file_path)

        fig = px.scatter(data, y='intensity', x='pm', template='plotly_white')
        fig.show()
    return render_template('predict_mgf.html', tables=[current_results], titles=[''])


class Predictor(object):
    def __init__(self, config, out_dir):
        self.out_dir = out_dir
        self.pipeline = PipelineProcessor(config)

    def predict_file(self, file_path):
        assert file_path
        assert self.pipeline
        steps = ['load from file', 'clean-load', 'scale-load', 'align', 'validate']
        self.pipeline.config.pipeline_steps = steps
        self.pipeline.config.file_path = file_path
        self.pipeline.config.save_output = True
        self.pipeline.config.out_dir = self.out_dir
        self.pipeline.process_pipeline()
        filename = os.path.basename(file_path)
        out_data = ModelOutData(filename=filename, out_dir=self.out_dir)
        assert out_data.out_path
        out_file = os.path.join(out_data.out_path, out_data.predict_data)
        data = pd.read_csv(out_file, index_col='File')
        return data

    def load_original(self, file_path):
        assert file_path
        filename = os.path.basename(file_path)
        out_data = ModelOutData(filename=filename, out_dir=self.out_dir)
        assert out_data.out_path
        out_file = os.path.join(out_data.out_path, out_data.original_data)
        data = pd.read_csv(out_file)
        return data

    def load_cleaned(self, file_path):
        assert file_path
        filename = os.path.basename(file_path)
        out_data = ModelOutData(filename=filename, out_dir=self.out_dir)
        assert out_data.out_path
        out_file = os.path.join(out_data.out_path, out_data.clean_data)
        data = pd.read_csv(out_file)
        return data

    def load_cam(self, file_path):
        assert file_path
        filename = os.path.basename(file_path)
        out_data = ModelOutData(filename=filename, out_dir=self.out_dir)
        assert out_data.out_path
        out_file_clean = os.path.join(out_data.out_path, out_data.clean_data)
        out_file_cam = os.path.join(out_data.out_path, out_data.cam_data)
        clean_data = pd.read_csv(out_file_clean)
        cam_data = pd.read_csv(out_file_clean)

        return data


if __name__ == '__main__':
    app.run(debug=True)
