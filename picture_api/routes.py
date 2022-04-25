from flask import jsonify, make_response, request, render_template
from flask import Flask
from picture_api import controllers

app = Flask(__name__)


@ app.route('/', methods=['GET', 'POST'])
def index():
    return controllers.index()


@ app.route('/capture', methods=['GET'])
def capture_images():
    return controllers.capture_images()


if __name__ == '__main__':
    app.run(port='5000')
