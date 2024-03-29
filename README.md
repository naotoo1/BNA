# BNA

End-to-end implementation of banknotes authentication using advanced prototype-based classification by components model within the flask framework for swagger API and streamlit app dockerized and deployed on Heroku platform as a service cloud

## What is it?
This project utilizes classification by components prototype-based model in the area of interpretable-ai to detect genuine and forged banknotes. The implementation uses ensemble techniques with options available for stand-alone models.

## How to use
To authenticate banknotes and return the confidence of the authentication, users have the following options to select from: ```Soft```, ```Hard```, ```Random``` and ```None```. 

If the option is soft or hard, an ensemble of trained ```cbc``` models with different configurations is utilized for the classification. If random, a trained ```cbc``` model is randomly selected from the ensemble to take over the classification. If none, the most confident trained ```cbc``` model out of the ensemble takes charge of the classification.

### To use BNA streamlit app deployed on Heroku,
1. click on the link https://bna-mlapp.herokuapp.com/
2. Follows steps 2 to 5 for the streamlit app version described below to authenticate banknotes.

### To run BNA flask-swagger api on a browser locally,
python 3.9 or later with all [requirements.txt](https://github.com/naotoo1/BNA/blob/main/requirements.txt) dependencies installed

```python
git clone https://github.com/naotoo1/BNA.git
cd BNA
pip install -r requirements.txt
```
Run
```python
python app1.py 
```

1. Get the local URL ```http://localhost/apidocs```
2. To authenticate a single test case for a banknote, click on ```Get``` ---> ```Try it out```
3. Enter the values for variance, skewness, curtosis and entropy with method chosen either as soft, hard, random or none (ie left blank)
4. click on ```Execute```
5. To authenticate a multiple test case for some banknotes, click on ```Post``` ---> ```Try it out```
6. Choose the ```csv``` containing test case data with the following designated features variance, skewness, curtosis and entropy
7. Enter the method either as soft, hard, random or none (ie left blank)
8. click on ```Execute```

### To run BNA streamlit app version on a browser locally,
Clone the repository and cd into the folder BNA as described above and  
Run
```python
streamlit run app.py
```
1. Get the local host URL ```http://localhost:8501```
2. To authenticate a single test case for a banknote, enter the values for variance, skewness, curtosis and entropy with the method chosen either as soft, hard, random    or none
3. click on Predict
4. To authenticate a multiple test case for some banknotes, click on browse files to choose the csv file containing test case data with the following designated          features variance, skewness, curtosis and entropy
5. click on Predict_file


### To build dockers image file for BNA and run in a browser,
1. Run ```docker build -t nameofapp .``` in cmd
2. Run ```docker run -p port:port nameofapp``` in cmd
3. Run ```http://localhost:port/apidocs```    





