FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-2

WORKDIR /app 

COPY . .
RUN pip install -r requirements.txt
RUN cd blob_env && pip install -e .