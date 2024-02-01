FROM python:3.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# timezone
RUN ln -snf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime && echo Asia/Ho_Chi_Minh > /etc/timezone

RUN apt-get update && apt-get install gcc wget fontconfig poppler-utils ffmpeg libsm6 libxext6 curl -y
RUN wget http://ftp.de.debian.org/debian/pool/non-free/f/fonts-ipafont-nonfree-jisx0208/fonts-ipafont-nonfree-jisx0208_00103-7_all.deb
RUN dpkg -i fonts-ipafont-nonfree-jisx0208_00103-7_all.deb
RUN fc-cache -f -v

COPY . /app
WORKDIR /app
ADD ./requirements.txt requirements.txt
# Install pip requirements
RUN python -m pip install pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install pdfminer.six --upgrade
RUN pip uninstall -y python-docx


RUN useradd -ms /bin/bash celery
RUN chown -R celery:celery /app
USER celery

CMD ["/bin/bash"]
