# Tensorflow_Template
my tensorflow project template

Folder structure main reference https://github.com/google/seq2seq and https://github.com/MrGemy95/Tensorflow-Project-Template


## Folder Structure
```
├── data_loader
│   └── data_generator.py   - here's the data_generator that is responsible for all data handling.
│
│
├── model                   - this folder contains any model of your project.
|   ├── configurable.py     - Interface for all classes that are configurable via a parameters dictionary.  
|   ├── mode_base.py        - Abstract base class for models.
│   └── example_model.py    - Example model
│
│
├── runner                  - this folder contains trainers and inferences of your project.
|   ├── runner_base.py      - Base class for model trainer and infer
│   |── example_trainer.py  - Example infer
│   └── example_trainer.py  - Example trainer
│ 
│── utils
│   └── misc_utils.py
│
|—— arguments.py            - Parse and load parameters function
|
└—— main.py                 - Main function
```
