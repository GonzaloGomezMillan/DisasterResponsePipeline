# **Disaster Response Pipeline**
This project is part of the Nanodegree Programm "Data Scientist" of Udacity. In this project I have build a machine learning model to categorize emergency messages based on the needs communicated by the sender.

---


## **Quick start**
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
---
## **File description**

- app
    - `run.py`: File which runs the web application


- data
    - `DisasterResponse.db`: Database where the prepared dataset is stored by process_data.py
    - `disaster_categories.csv`: Raw data set with categories of the messages.
    - `disaster_messages.csv`: Raw data set with the messages used for the analysis
    - `process_data.py`: Python file which is in charge of preparing the dataset and saving it in a SQL database to be processed later by the ETL Pipeline. 

- models
    - `train_classifier_py`: Python file which uses the prepared dataset to train the machine learning model trough an ETL Pipeline. 

---

## **Copyright and license**
**Author**: The author of the data analysis, data preparation, machine learning model and  web app is Gonzalo Gomez Millan, based on the proposed Udacity Project.

**Acknowledgments**: Also noteworthy is the work of Udacity facilitating the environment to develop this project, which is part of the Data Scientist Nanodegree Program.

![Udacity](https://upload.wikimedia.org/wikipedia/commons/e/e8/Udacity_logo.svg)



