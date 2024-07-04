# Disaster Response Pipeline Project (Udacity - Data Scientist Nanodegree Program)
## Table of Contents
1. [Introduction](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#introduction)
2. [File Descriptions](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#file-descriptions)
3. [Installation](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#installation)
4. [Instructions](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#instructions)
5. [Acknowledgements](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#acknowledgements)
6. [Screenshots](https://github.com/Raoul-Batcho/udacity-disaster-response-pipeline#screenshots)

## Introduction
In this Project Workspace, we work with real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code! [Figure Eight](https://www.figure-eight.com/).

## File Descriptions
### Folder: app
**run.py** - python script to launch web application.<br/>
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

### Folder: data
**disaster_messages.csv** - real messages sent during disaster events (provided by Figure Eight)<br/>
**disaster_categories.csv** - categories of the messages<br/>
**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database<br/>
**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ETL pipeline<br/>
**DisasterResponse.db** - cleaned data stored in SQlite database

### Folder: models
**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use<br/>
**classifier.pkl** - pickle file contains trained model<br/>
**ML Pipeline Preparation.ipynb** - Jupyter Notebook used to prepare ML pipeline

## Installation
There should be no extra libraries required to install apart from those coming together with Anaconda distribution. There should be no issue to run the codes using Python 3.5 and above.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
* [Udacity](https://www.udacity.com/) for providing an excellent Data Scientist training program.
* [Figure Eight](https://www.figure-eight.com/) for providing dataset to train our model.

## Screenshots
1. Main page shows the Overview of Training Dataset & Distribution of Message Categories
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/1.%20main%20page.JPG)

2. After entering the message and clicking 'Classify Message', we can see the category(ies) of which the message is classified to , highlighted in green
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/3.%20classify%20result.JPG)

3. Run process_data.py for ETL pipeline
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/4.%20disaster_img.png)

4. Run train_classifier.py for ML pipeline
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/5.%20disaster_img.png)
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/6.%20disaster_img.png)
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/7.%20disaster_img.png)
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/8.%20disaster_img.png)
5. Run run.py in app's directory to run web app<br/>
![image](https://github.com/Raoul-Batcho/Udacity-Disaster-Response-Pipeline/blob/main/screnshots/9%20disaster_img.png)
