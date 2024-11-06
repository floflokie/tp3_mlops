import requests

# test the /predict endpoint
def test_predict():
    url = "http://127.0.1:5000/predict"
    data = [[5.1, 3.5, 1.4, 0.2]]
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}
test_predict()
# test the /update-model endpoint
def test_update_model():
    url = "http://127.0.1:5000/update-model"
    data = {"model_name": "iris_model", "model_version": "1"}
    response = requests.post(url, json=data)
    print(response.json())
    assert response.status_code == 200

test_update_model()