FROM python:3.6.8-slim

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev

WORKDIR /
COPY src /src/
COPY app.py /app.py
COPY download_model.py /download_model.py

RUN pip install iterative-stratification==0.1.6 \ 
                numpy==1.16.0 \
                Pillow==6.2.0 \
                torch==1.3.0 \
                torchvision==0.4.1 \
                pandas==0.25.1 \
                imgaug==0.3.0 \
                scikit-learn==0.19.0 \
                opencv-python==4.1.1.26 \
                efficientnet-pytorch==0.5.0 \
                tqdm==4.40.2 \ 
                Flask==1.1.1 \
                flask-cors==3.0.7 \
                gunicorn==19.9.0 

RUN python download_model.py

CMD ["gunicorn", "-b", "0.0.0.0:2000", "app"]
