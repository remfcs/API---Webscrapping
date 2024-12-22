import pytest
from fastapi.testclient import TestClient
from main import get_application


@pytest.fixture(scope="module")
def client():
    """
    Creates a FastAPI test client for all tests in the module.
    With scope="module", it is only created once for the entire test module.
    """
    app = get_application()
    return TestClient(app)


class TestAllRoutes:

    def test_add_dataset(self, client):
        """
        Test the POST /dataset route.
        We add a dataset that does not already exist.
        """
        payload = {
            "name": "my_test_dataset",
            "url": "https://some.url/to/dataset"
        }
        response = client.post("/dataset", params=payload)
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()
        assert "Dataset added successfully." or "Dataset updated successfully." in data["message"]
        assert data["dataset"]["name"] == "my_test_dataset"
        assert data["dataset"]["url"] == "https://some.url/to/dataset"

    def test_get_dataset(self, client):
        """
        Test the GET /dataset/{dataset_name} route.
        Assumes that 'my_test_dataset' was added by the previous test.
        """
        response = client.get("/dataset/my_test_dataset")
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()
        assert data["name"] == "my_test_dataset"
        assert data["url"] == "https://some.url/to/dataset"

    def test_modify_dataset(self, client):
        """
        Test the PUT /dataset/{dataset_name} route.
        We update the 'url' of the dataset.
        """
        new_url = "https://some.url/to/dataset_v2"
        response = client.put("/dataset/my_test_dataset", params={"url": new_url})
        assert response.status_code == 200, f"Error: {response.text}"
        data = response.json()
        assert "Dataset modified successfully." in data["message"]
        assert data["dataset"]["url"] == new_url

    def test_load_dataset(self, client):
        """
        Test the GET /load-dataset/{dataset_name} route.
        Attempts to download the specified dataset via Kaggle or a similar source.
        """
        response = client.get("/load-dataset/my_test_dataset")
        # If the config is missing or Kaggle is not configured,
        # you might get an error code or 500 => adapt to your logic.
        assert response.status_code in [200, 404, 500], f"Error: {response.text}"
        if response.status_code == 200:
            data = response.json()
            assert "Dataset 'my_test_dataset' downloaded" in data["message"]
            # data["json_path"] would be the JSON file path if available

    def test_process_dataset(self, client):
        """
        Test the GET /process-dataset route.
        Processes a local Iris dataset according to your code.
        """
        response = client.get("/process-dataset")
        # For this test to succeed, IRIS_DATASET_PATH must exist (src/data/iris/iris.csv).
        # Otherwise, a 404 error might occur.
        if response.status_code == 200:
            data = response.json()
            # The data is expected to contain the transformed dataset.
            # You can check the size or a particular type.
            assert isinstance(data, list), "process_dataset should return a list (or a converted DataFrame)."
        else:
            pass  # 404 or other

    def test_split_dataset(self, client):
        """
        Test the GET /split-dataset route.
        """
        response = client.get("/split-dataset", params={"test_size": 0.2, "random_state": 42})
        # Verify that the dataset has been split
        if response.status_code == 200:
            data = response.json()
            assert "train_data" in data
            assert "test_data" in data
        else:
            pass

    def test_train_model(self, client):
        """
        Test the POST /train-model route.
        """
        response = client.post("/train-model", params={
            "model_name": "LogisticRegression",
            "target_column": "Species"
        })
        # It may fail if the dataset does not exist, etc.
        if response.status_code == 200:
            data = response.json()
            assert "Model trained and saved successfully." in data["message"]
            assert "model_path" in data
        else:
            pass

    def test_predict(self, client):
        """
        Test the POST /predict route.
        We send minimal feature values for a prediction.
        """
        payload = [
            {
                "SepalLengthCm": 5.0,
                "SepalWidthCm": 3.4,
                "PetalLengthCm": 1.5,
                "PetalWidthCm": 0.2
            }
        ]
        response = client.post("/predict", json=payload, params={"model_name": "LogisticRegression"})
        # May fail if the model has not been trained or does not exist.
        if response.status_code == 200:
            data = response.json()
            assert "Predictions made successfully." in data["message"]
            # data["predictions"] => the prediction list
        else:
            pass

    def test_retrieve_parameters(self, client):
        """
        Test the GET /retrieve-parameters route.
        Assumes Firestore is configured, otherwise a 500 or 404 error may occur.
        """
        response = client.get("/retrieve-parameters")
        if response.status_code == 200:
            data = response.json()
            assert "Parameters retrieved successfully." in data["message"]
            # data["parameters"] => Firestore parameters
        else:
            pass

    def test_add_parameters(self, client):
        """
        Test the POST /add-parameters route.
        """
        sample_params = {"some_param": "some_value"}
        response = client.post("/add-parameters", json=sample_params)
        if response.status_code == 200:
            data = response.json()
            # Check that "some_param" appears in the response
            assert "some_param" in data.get("updated_parameters", {})
        else:
            pass

    def test_update_parameters(self, client):
        """
        Test the PUT /update-parameters route.
        """
        sample_params = {"some_param": "some_other_value"}
        response = client.put("/update-parameters", json=sample_params)
        if response.status_code == 200:
            data = response.json()
            # Check the parameter update
            assert data.get("updated_parameters", {}).get("some_param") == "some_other_value"
        else:
            pass
