from picture_api.utils.image_utils import capture_image_batch, find_last_images
from flask import jsonify, make_response, request, render_template, send_from_directory, abort
from flask import Flask
import os

def capture_images():
    data = request.args
    batch_size = int(data.get("batchSize", 10))
    mono_camera_index_left = int(data.get("camIndexLeft", 1))
    mono_camera_index_right = int(data.get("camIndexRight", 2))
    status, capure_arr = capture_image_batch(
        batch_size, "./left", "./right", mono_camera_index_left, mono_camera_index_right)
    return make_response(jsonify({'status': status,
                                  "statusArray": capure_arr}), 200)


def index():
    if request.method == 'POST':
        data = request.form
        batch_size = int(data.get("batchSize", 10))
        mono_camera_index_left = int(data.get("camIndexLeft", 1))
        mono_camera_index_right = int(data.get("camIndexRight", 2))
        status, capure_arr = capture_image_batch(
            batch_size, "./left", "./right", mono_camera_index_left, mono_camera_index_right)
        return render_template('capture_form.html',
                               form=request.form,
                               arr=capure_arr,
                               status=status)

    elif request.method == 'GET':
        return render_template('capture_form.html',
                               form=request.form)

def images(dir_left:str, dir_right:str):
    data = request.args
    
    image_str_left, image_str_right = find_last_images(dir_left, dir_right)
    
    if data.get('cam')=='right' and image_str_right:
        return send_from_directory(os.path.join(os.getcwd(),dir_right),image_str_right)
    elif data.get('cam')=='left' and image_str_left:
        return send_from_directory(os.path.join(os.getcwd(),dir_left),image_str_left)
    else:
        abort(404)