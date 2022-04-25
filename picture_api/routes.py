from flask import jsonify, make_response, request
from flask import Flask
from utils.image_utils import capture_image_batch

app = Flask(__name__)


@ app.route('/', methods=['GET'])
def root_route():
    return "<h4>Hello, you are using picture-api!</h4>"


@ app.route('/capture', methods=['GET'])
def capture_images():
    data = request.args
    batch_size = int(data.get("batchSize", 10))
    print(batch_size)
    mono_camera_index = int(data.get("camIndex", 2))
    status, capure_arr = capture_image_batch(
        batch_size, "./mono_camera_images", "./point_cloud", mono_camera_index)
    return make_response(jsonify({'status': status,
                                  "statusArray": capure_arr}), 200)


if __name__ == '__main__':
    app.run(port='5000')
