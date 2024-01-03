FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /home

# Install pip
RUN apt-get update
RUN apt-get install -y python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

LABEL envs="[{\"name\":\"TRAINING_DATA_PATH\",\"type\":\"text\",\"required\":true}, \
    {\"name\":\"VALIDATION_DATA_PATH\",\"type\":\"text\",\"required\":true}, \
    {\"name\":\"TRAINING_PARTITION_FILE\",\"type\":\"text\",\"required\":false},\
    {\"name\":\"VALIDATION_PARTITION_FILE\",\"type\":\"text\",\"required\":false}, \
    {\"name\":\"CLIENT_ID\",\"type\":\"number\",\"required\":true}, \
    {\"name\":\"DEVICE\",\"type\":\"text\",\"required\":false}, \
    {\"name\":\"NUM_EPOCHS\",\"type\":\"number\",\"required\":false}]"

CMD [ "python", "./main.py" ]
