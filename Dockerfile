FROM python:3.7
COPY . /main
WORKDIR /main
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 -bind 0.0.0.0:$PORT main:app