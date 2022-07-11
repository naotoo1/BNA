import torch
from prosemble import Hybrid
import torch.utils.data
import numpy as np
import pandas as pd
import streamlit as st

# load some prototype based models
model1 = torch.load('cbc.pt')  # zeros init(1pc)
model2 = torch.load('cbc1.pt')  # feature init(1pc) with noise(0.1)
model3 = torch.load('cbc2.pt')  # feature init(1pc) with noise(0.2)

# prototype labels
proto_classes = np.array([0, 1])

# Instantiate ensemble from prosemble
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')

# Instantiate the Hybrid class
model1_ = Hybrid(model_prototypes=model1.prototypes.detach().numpy(), proto_classes=proto_classes, mm=2,
                 omega_matrix=None,
                 matrix='n')
model2_ = Hybrid(model_prototypes=model2.prototypes.detach().numpy(), proto_classes=proto_classes, mm=2,
                 omega_matrix=None,
                 matrix='n')
model3_ = Hybrid(model_prototypes=model3.prototypes.detach().numpy(), proto_classes=proto_classes, mm=2,
                 omega_matrix=None,
                 matrix='n')


# get confidence of test file
def get_conf(x, y):
    return [[i[j][y] for j in range(len(i))] for i in x]


# get sorted confidence for test file
def get_sorted_conf(x, y):
    return [[i[j] for i in x] for j in range(y)]


# get index of max confidence prediction
def get_argmax(x):
    return [np.argmax(conf) for conf in x]


#  get confidence of most confident model
def get_max_conf(x):
    return [np.max(conf) for conf in x]


# get pred of most confident model
def get_conf_pred(x, y):
    return [y[i][x[i]] for i in range(len(x))]


# choose random_model
def get_index_random_model(x):
    return np.random.choice(range(x))


# get the pred and confidence for the random model
def get_random_conf(x, y):
    return y[x]


#  get security of most confident model
def get_sec_(x):
    return [i.flatten()[1] for i in x]


# get index of max confidence
def get_sec_index_(x):
    return np.argmax(x)


def get_max_sec_(x):  # r__ uses r
    return np.max(x)


def welcome():
    return "Bank Note Authenticator"


def predict_note(variance, skewness, curtosis, entropy, method):
    """Let's Authenticate the Banks Note
        This is using docstrings for specifications.
        ---
        parameters:
          - name: variance
            in: query
            type: number
            required: true
          - name: skewness
            in: query
            type: number
            required: true
          - name: curtosis
            in: query
            type: number
            required: true
          - name: entropy
            in: query
            type: number
            required: true
          - name: method
            in: query
            type: string
            required: true
        responses:
            200:
                description: The output values

        """

    X_test = [[variance, skewness, curtosis, entropy]]
    X_test = [float(y) for x in X_test for y in x]

    # Get of confidence of predicted results with optimised security_coef chosen as 2
    sec1 = model1_.get_security(x=np.array([X_test]), y=2)
    sec2 = model2_.get_security(x=np.array([X_test]), y=2)
    sec3 = model3_.get_security(x=np.array([X_test]), y=2)

    # Get prediction from the prototype models
    prediction1 = model1.predict(torch.Tensor([X_test]))
    prediction2 = model2.predict(torch.Tensor([X_test]))
    prediction3 = model3.predict(torch.Tensor([X_test]))

    all_pred = [prediction1, prediction2, prediction3]
    all_sec = [sec1, sec2, sec3]
    # prediction from the ensemble using hard voting
    prediction11 = ensemble.pred_prob(np.array([[variance, skewness, curtosis, entropy]]), all_pred)
    # prediction from the ensemble using soft voting
    prediction22 = ensemble.pred_sprob(np.array([[variance, skewness, curtosis, entropy]]), all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob(np.array([[variance, skewness, curtosis, entropy]]), all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob(np.array([[variance, skewness, curtosis, entropy]]), all_sec)

    # prediction and confidence from a randomly chosen model in the ensemble
    rand_conf = get_random_conf(x=get_index_random_model(x=3), y=all_sec).flatten()

    # index of max confidence
    index = get_sec_index_(x=get_sec_(x=all_sec))  # r_

    # confidence from most confident model
    conf = get_max_sec_(x=get_sec_(x=all_sec))  # r__

    if method == 'soft':
        if prediction22[0] > 0.5:
            return f"Counterfeit with {round(soft_prob[0] * 100, 2)}% confidence"
        else:
            return f" Original with {round(soft_prob[0] * 100, 2)}% confidence"

    if method == 'hard':
        if prediction11[0] > 0.5:
            return f"Counterfeit with {round(hard_prob[0] * 100, 2)}% confidence"
        else:
            return f"Original with {round(hard_prob[0] * 100, 2)}% confidence"

    if method == 'random':
        if rand_conf[0] > 0.5:
            return f"Counterfeit with {round(rand_conf[1] * 100, 2)}% confidence"
        else:
            return f"Original with {round(rand_conf[1] * 100, 2)}% confidence"

    if method == 'None':
        pred = all_pred[index][0]
        if pred > 0.5:
            return f"Counterfeit with {round(conf * 100, 2)}% confidence"
        else:
            return f"Original with {round(conf * 100, 2)}% confidence"


def predict_BankNoteFile(file, method):
    """
    predict BankNote file
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      - name: method
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
    """
    df = file.to_numpy()
    prediction1 = model1.predict(torch.Tensor(df))
    prediction2 = model2.predict(torch.Tensor(df))
    prediction3 = model2.predict(torch.Tensor(df))

    sec1 = model1_.get_security(x=df, y=2)
    sec2 = model2_.get_security(x=df, y=2)
    sec3 = model3_.get_security(x=df, y=2)

    all_pred = [prediction1, prediction2, prediction3]
    all_sec = [sec1, sec2, sec3]

    # prediction from the ensemble using hard voting
    prediction11 = ensemble.pred_prob(df, all_pred)
    # prediction from the ensemble using soft voting
    prediction22 = ensemble.pred_sprob(df, all_sec)
    # confidence of the prediction using hard voting
    hard_prob = ensemble.prob(df, all_pred)
    # confidence of the prediction using soft voting
    soft_prob = ensemble.sprob(df, all_sec)

    # prediction and confidence from a randomly chosen model in the ensemble
    rand_conf = get_random_conf(x=get_index_random_model(x=3), y=all_sec)

    num = len(df)

    # predictions from the most confident model in terms of recall procedure in the ensemble
    pred1 = get_conf_pred(x=get_argmax(get_sorted_conf(get_conf(x=all_sec, y=1), y=num)),
                          y=get_sorted_conf(x=get_conf(x=all_sec, y=0), y=num))

    # confidence from the the most confident model in the ensemble
    conf1 = get_max_conf(x=get_sorted_conf(x=get_conf(x=all_sec, y=1), y=num))

    list_ = []

    if method == 'soft':
        for i, p in enumerate(prediction22):
            if p > 0.5:
                list_.append(f" {i}. Counterfeit with {round(soft_prob[i] * 100, 2)}% confidence")
            else:
                list_.append(f" {i}. Original with {round(soft_prob[i] * 100, 2)}% confidence")
        pred = np.array(list_)
        pred = pred.reshape(pred.size, 1)
        pred = ''.join(str(x) for x in pred)

        return pred

    if method == 'hard':
        for i, p in enumerate(prediction11):
            if p > 0.5:
                list_.append(f" {i}. Counterfeit with {round(hard_prob[i] * 100, 2)}% confidence")
            else:
                list_.append(f" {i}. Original with {round(hard_prob[i] * 100, 2)}% confidence")
        pred = np.array(list_)
        pred = pred.reshape(pred.size, 1)
        pred = ''.join(str(x) for x in pred)
        return pred

    if method == 'random':
        for i, p in enumerate(rand_conf):
            if p[0] > 0.5:
                list_.append(f" {i}. Counterfeit with {round(p[1] * 100, 2)}% confidence")
            else:
                list_.append(f" {i}. Original with {round(p[1] * 100, 2)}% confidence")
        pred = np.array(list_)
        pred = pred.reshape(pred.size, 1)
        pred = ''.join(str(x) for x in pred)
        return pred

    if method == 'None':
        for i, p in enumerate(pred1):
            if p > 0.5:
                list_.append(f" {i}. Counterfeit with {round(conf1[i] * 100, 2)}% confidence")
            else:
                list_.append(f" {i}. Original with {round(conf1[i] * 100, 2)}% confidence")
        pred = np.array(list_)
        pred = pred.reshape(pred.size, 1)
        pred = ''.join(str(x) for x in pred)
        return pred


def main():
    st.title("Bank Note Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Note Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("skewness", "Type Here")
    curtosis = st.text_input("curtosis", "Type Here")
    entropy = st.text_input("entropy", "Type Here")
    method = st.text_input("method", "Type Here")
    result = ""
    dataframe = ""
    results = ""

    if st.button("Predict"):
        result = predict_note(variance, skewness, curtosis, entropy, method)
    st.success('The output is {}'.format(result))
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    if st.button('predict_file'):
        results = predict_BankNoteFile(dataframe, method)
    st.success('The output is {}'.format(results))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
