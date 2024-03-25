## Traballo de Aprendizaxe Automatico I

### Instalacion en linux
	
	git clone https://gitlab.com/honsalosoeh/emotions-classifier.git
	cd emotions-classifiers
	pip install -r requirements.txt
	kaggle datasets download -d sujaykapadnis/emotion-recognition-dataset
	unzip emotion-recognition-dataset.zip; rm emotion-recognition-dataset.zip
	python traballo.py
