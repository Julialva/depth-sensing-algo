from picture_api.utils.image_utils import capture_image_batch
from flask import jsonify, make_response, request, render_template
from flask import Flask


def capture_images():
    data = request.args
    batch_size = int(data.get("batchSize", 10))
    mono_camera_index = int(data.get("camIndex", 2))
    status, capure_arr = capture_image_batch(
        batch_size, "./mono_camera_images", "./point_cloud", mono_camera_index)
    return make_response(jsonify({'status': status,
                                  "statusArray": capure_arr}), 200)


def index():
    data = request.form
    if request.method == 'POST':
        batch_size = int(data.get("batchSize", 10))
        mono_camera_index = int(data.get("camIndex", 2))
        status, capure_arr = capture_image_batch(
            batch_size, "./mono_camera_images", "./point_cloud", mono_camera_index)
        return render_template('capture.html',
                               form=request.form,
                               arr=capure_arr,
                               status=status)

    elif request.method == 'GET':
        return render_template('capture.html',
                               form=request.form)
