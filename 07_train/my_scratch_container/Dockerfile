FROM python:3.8-slim-buster

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++

RUN pip3 install pandas numpy joblib scikit-learn

# ADD train /opt/ml/code/train
ADD my_scratch_container/trainer.py ./trainer.py

WORKDIR /

ENTRYPOINT ["python", "trainer.py"]