FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD CNN_script.py .
ADD calibrator.py .
COPY final_pics/ .
COPY calib/ .
COPY calib/ .
ENTRYPOINT ["python","-u", "CNN_script.py"]