FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN ["apt-get", "update"]
#RUN ["apt-get", "install", "-y", "zsh"]
RUN ["apt-get", "install", "-y", "git"]
RUN ["apt-get", "update"]
#RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
#RUN pip install torch==1.11.0+cu115 --find-links https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --upgrade
WORKDIR /workspace/repositories/XIConceptLearning/
ENV PYTHONPATH "${PYTHONPATH}:./"