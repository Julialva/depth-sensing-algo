FROM tensorflow/tensorflow:latest-gpu
ENV DIR=/depth
WORKDIR /depth
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.txt ${DIR}
RUN pip install -r requirements.txt
ADD Final_pics.zip ${DIR}
ADD CNN_script.py ${DIR}
ADD calibrator.py ${DIR}
ADD calib/right_cal/* ${DIR}/calib/right_cal/
ADD calib/left_cal/* ${DIR}/calib/left_cal/
VOLUME ["./depth"]
ENTRYPOINT ["python","-u", "CNN_script.py"]