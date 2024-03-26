## Trabajo de Aprendizaxe Automatico I

### Instalacion en linux
	
	git clone https://github.com/jmorales5162/emotiones_classifier_hub.git
	cd emotiones_classifier_hub
	pip install -r requirements.txt
	kaggle datasets download -d sujaykapadnis/emotion-recognition-dataset
	unzip emotion-recognition-dataset.zip; rm emotion-recognition-dataset.zip
	python main.py
