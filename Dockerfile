FROM docker.io/entilzha/quizbowl:0.1

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN conda activate qa
RUN python -m nltk.downloader punkt