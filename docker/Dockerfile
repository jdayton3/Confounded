FROM tensorflow/tensorflow:1.11.0-py3
WORKDIR /
ARG project="confounded"
ARG destination="/${project}"
RUN mkdir ${destination} && \
    #curl -sL https://github.com/jdayton3/Confounded/archive/docker-0.1.0.tar.gz | \
    curl -sL https://github.com/jdayton3/Confounded/archive/master.tar.gz | \
    tar xz -C ${destination} --strip-components 1
WORKDIR ${destination}
RUN echo '#!/bin/bash\ncd /confounded\npython -m src.autoencoder "$@"' > /usr/bin/confounded && \
    chmod +x /usr/bin/confounded
ENTRYPOINT [ "/usr/bin/confounded" ]
