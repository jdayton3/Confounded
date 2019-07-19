FROM tensorflow/tensorflow:1.11.0-py3
RUN pip install --upgrade pip && pip install tqdm
COPY ./confounded /confounded/confounded
COPY ./data/bladderbatch/bladderbatch.csv /confounded/data/bladderbatch/bladderbatch.csv
RUN mkdir -p /confounded/data/metrics
WORKDIR /confounded
RUN echo '#!/bin/bash\ncd /confounded\npython -m confounded "$@"' > /usr/bin/confounded && \
    chmod +x /usr/bin/confounded
ENTRYPOINT [ "/usr/bin/confounded" ]
