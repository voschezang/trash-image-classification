start:
	jupyter notebook src/

startw:
	pythonw src/main.py

logs:
	tensorboard --logdir=/tmp/ml_model

dev:
	python3 src/main.py

install:
	pip3 install -r requirements.txt

install2:
	pip install -r requirements.txt

clean:
	find . -name \*.pyc -delete

cprofile:
	python3 -m cProfile -o test/program.prof src/main.py

snakeviz_:
	snakeviz test/program.prof
