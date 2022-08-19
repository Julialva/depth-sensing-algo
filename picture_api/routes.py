from flask import jsonify, make_response, request, render_template, url_for

from flask import Flask
from picture_api import controllers


app = Flask(__name__)
app.config['LEFT_IMAGES'] = './left'
app.config['RIGHT_IMAGES'] = './right'


@ app.route('/', methods=['GET', 'POST'])
def index():
    return controllers.index()


@ app.route('/capture', methods=['GET'])
def capture_images():
    return controllers.capture_images()

@app.route('/last_images')
def last_images():
    return controllers.images(app.config['LEFT_IMAGES'], app.config['RIGHT_IMAGES'])

@app.route('/show_last_images')
def show_last_images():
    left_file_url = url_for('last_images',cam='left')
    right_file_url = url_for('last_images',cam='right')
    return render_template('captured.html',left_file_url=left_file_url, right_file_url=right_file_url)
    