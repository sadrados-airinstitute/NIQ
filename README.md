# NIQ

# NIQ - Nutritional information extraction from product images

This repository is structured to solve a two-part task aimed at the exploration and modeling of data derived from **Open Food Facts**, a large-scale public database with multimodal product information.


## Project folder structure

```plaintext

NIQ
├── api.py
├── data/
├── data_preprocessing/
│   ├── __init__.py
│   ├── image_preprocessing.py
│   └── text_preprocessing.py
├── logs/
│   └── api.log
├── model/
│   ├── __init__.py
│   ├── classifier_model.py
│   ├── entity_recognition_model.py
│   ├── evaluation.py
│   ├── nutrition_extraction_pipeline.py
│   ├── ocr_model.py
│   └── train.py
├── models/
│   └── prueba.pth
├── notebooks/
│   └── EDA.ipynb
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── create_dataset.py
│   ├── logger.py
│   └── visualization.py
├── main.py
├── NIQ.drawio
├── README.md
└── requirements.txt
