FROM gcr.io/kubeflow-images-public/tensorflow-2.1.0-notebook-cpu

COPY . .
RUN pip install -r requirements.txt
