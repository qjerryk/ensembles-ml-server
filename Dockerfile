FROM python:3.10.1

COPY --chown=root:root app /root/app/
COPY --chown=root:root ensembles /root/app/ensembles

WORKDIR /root/app

RUN pip3 install -r requirements.txt
RUN chmod +x run.py

ENV SECRET_KEY reallysecret

CMD ["python", "run.py"]