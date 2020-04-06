# Q&A System - StackOverflow Assistant Bot
Build a dialog chatbot that be able to answer programming-related questions using StackOverflow dataset.

**Build the container image**
```shell
$ docker build -t bot .
```

**Run the container**
```shell
$ docker run -it --name bot  -p 8080:8080 -v $PWD:/root/bot bot 
```

Jupyter notebook will start automatically at `your_ip:8080`.

**Launch app**
```shell
$ python main.py
``` 
