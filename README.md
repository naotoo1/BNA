# BNA

End-to-end implementation of advanced prototype-based classification by components model for the detection of genuine and forged bank notes within the flask framework for swagger API and streamlit app with virtualization and containerization in dockers with deployment on Heroku platform as a service cloud

## What is it?
This project utilizes classification by components prototype-based model in the area of interpretable-ai to detect genuine and forged bank notes. The implementation uses ensemble techniques with options available for stand-alone models.

## How to use
To authenticate bank notes and return the confidence of the authentication, users have the following options to select from: ```Soft```, ```Hard```, ```Random``` and ```None```. 

If the option is soft or hard, an ensemble of trained ```cbc``` models with different configurations is utilized for the classification. If random, a trained ```cbc``` model is randomly selected from the ensemble to take over the classification. If none, the most confident trained ```cbc``` model out of the ensemble takes charge of the classification.

### To use BNA streamlit app deployed on Heroku,
1. click on the link https://bna-mlapp.herokuapp.com/
2. Follows steps 2 to 5 for the streamlit app version described below to authenticate bank notes.

### To run flask-swagger api on a browser,
1. Run ```python app1.py``` and get the local URL ```http://localhost/apidocs```
2. To authenticate a single test case for a bank note, click on ```Get``` ---> ```Try it out```
3. Enter the values for variance, skewness, curtosis and entropy with method chosen either as soft, hard, random or none (ie left blank)
4. click on ```Execute```
5. To authenticate a multiple test case for some bank notes, click on ```Post``` ---> ```Try it out```
6. Choose the ```csv``` containing test case data with the following designated features variance, skewness, curtosis and entropy
7. Enter the method either as soft, hard, random or none (ie left blank)
8. click on ```Execute```


### To run streamlit app version on a browser,
1. Run streamlit ```run app.py``` and get the local host URL ``` http://localhost:8501```
2. To authenticate a single test case for a bank note, enter the values for variance, skewness, curtosis and entropy with method chosen either as soft, hard, random or    none
3. click on ```Predict```
4. To authenticate a multiple test case for some bank notes, click on ```browse files``` to choose the ```csv``` containing test case data with the following designated features variance, skewness, curtosis and entropy
5. click on ```Predict_file``` 

### To build dockers image file for BNA and run in a browser,
1. Run ```docker build -t nameofapp .``` in cmd
2. Run ```docker run -p port:port nameofapp``` in cmd
3. Run ```http://localhost:port/apidocs```    





