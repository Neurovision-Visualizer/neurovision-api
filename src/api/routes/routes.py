from pyexpat import model
import pandas as pd
from api import app
from flask import json, request, jsonify, g, make_response, abort, redirect
from werkzeug.utils import secure_filename
from flask.helpers import send_from_directory
from ..controllers.DataHandler import DataHandler
from ..controllers.ModelHandler import ModelHandler
from ..util import dataset_util
from ..controllers.prep import prepHRT, prepHPR, prepIRS


@app.route("/api/data", methods=["GET"])
def get_data():
    dataset_names = ['heart_disease', 'house_price', 'iris']

    if request.method == "GET":
        # Retrieve and initialize dataset based on request
        selection = request.args.get('dataset')
        if selection not in dataset_names:
            return jsonify({"error": f"Invalid Dataset Entered: {selection}"}), 400

        # Create dataframe for dataset
        dataset = dataset_util.initialize_dataset(selection)

        # Create data handler
        datahandler = DataHandler(dataset)

        # Convert the dataset to JSON to be sent
        result = datahandler.toJSON()
        return jsonify({"msg": "Data Transmission Successful", "dataset": result}), 200

    return jsonify({"msg": "Method not Allowed"}), 405


"""
This route deals with the removal of a undesirable feature in a dataset

# Example Request
# {
#    "data": Dataset
#    "prob": "heart_disease", 
#    "featureName": "Species",
# }
"""


@app.route("/api/data/rfeature", methods=["POST"])
def remove_feature():
    if request.method == 'POST':
        try:
            # Reformat Data from request to Dataframe
            dataset = dataset_util.initialize_from_json(request.json["data"])

            # Create DataHandler with Dataframe
            datahandler = DataHandler(dataset)

            # Removal of Feature from Dataset
            datahandler.removeFeature(request.json["featureName"])

            # Return Dataset
            result = datahandler.toJSON()
            return jsonify({"msg": "Feature has been removed succesfully", "dataset": result}), 200
        except Exception as e:
            return jsonify({"msg": "An Internal Error Has Occured"}), 500
    return jsonify({"msg": "Method not Allowed"}), 405


"""
This route deals with the removal of invalid data

# Example Request
# {
#    "data": Dataset
#    "prob": "heart_disease", 
# }
"""


@app.route("/api/data/rinvalid", methods=["POST"])
def remove_invalid():
    if request.method == 'POST':
        try:
            # Reformat Data from request to Dataframe
            dataset = dataset_util.initialize_from_json(request.json["data"])

            # Create DataHandler with Dataframe
            datahandler = DataHandler(dataset)

            # Removal of Invalid Dataset Values
            datahandler.removeInvalidData()

            # Return Dataset
            result = datahandler.toJSON()
            return jsonify({"msg": "Invalid Data has been removed succesfully", "dataset": result}), 200
        except Exception as e:
            return jsonify({"msg": "An Internal Error Has Occured"}), 500
    return jsonify({"msg": "Method not Allowed"}), 405


"""
This route deals with the translation of the data

# Example Request
# {
#    "data": Dataset
#    "prob": "heart_disease", 
# }
"""


@app.route("/api/data/translate", methods=["POST"])
def translate_data():
    if request.method == 'POST':
        try:
            # Reformat Data from request to Dataframe
            dataset = dataset_util.initialize_from_json(request.json["data"])

            # Create DataHandler with Dataframe
            datahandler = DataHandler(dataset)

            # Translate Dataset Values
            datahandler.translateData()

            # Return Dataset
            result = datahandler.toJSON()
            return jsonify({"msg": "Features have been tanslated succesfully", "dataset": result}), 200
        except Exception as e:
            return jsonify({"msg": "An Internal Error Has Occured"}), 500
    return jsonify({"msg": "Method not Allowed"}), 405


"""
This route deals with the normalization of the data

# Example Request
# {
#    "data": Dataset
#    "prob": "heart_disease", 
# }
"""


@app.route("/api/data/normalize", methods=["POST"])
def normalize_data():
    if request.method == 'POST':

        # Reformat Data from request to Dataframe
        dataset = dataset_util.initialize_from_json(request.json["data"])

        # Create DataHandler with Dataframe
        datahandler = DataHandler(dataset)

        # Normalize Dataset Values with {}
        datahandler.normalizeData()

        # Return Dataset
        result = datahandler.toJSON()
        return jsonify({"msg": "Dataset has been normalized", "dataset": result}), 200

    return jsonify({"msg": "Method not Allowed"}), 405


"""
Developer - Backdoor
This route goes through all the phases of feature extraction to present 
the final dataset for Model Building
"""


@app.route("/api/data/qprep", methods=["GET"])
def get_prep():
    dataset_names = ['heart_disease', 'house_price', 'iris']

    if request.method == "GET":
        # Retrieve and initialize dataset based on request
        selection = request.args.get('dataset')
        if selection not in dataset_names:
            return jsonify({"error": f"Invalid Dataset Entered: {selection}"}), 400

        # Create dataframe for dataset
        dataset = dataset_util.initialize_dataset(selection)

        # Create data handler
        datahandler = DataHandler(dataset)

        if selection == 'heart_disease':
            datahandler = prepHRT(datahandler)
        elif selection == 'house_price':
            datahandler = prepHPR(datahandler)
        elif selection == 'iris':
            datahandler = prepIRS(datahandler)

        # Convert the dataset to JSON to be sent
        result = datahandler.toJSON()
        return jsonify({"msg": "Data Transmission and Prep Successful", "dataset": result}), 200

    return jsonify({"msg": "Method not Allowed"}), 405


"""
This route deals with the creation of the model based on the structure created by the user
It returns the training history as well as the evaluation metrics to be displayed on the frontend

# Example Request
# {
#    "data": Dataset
#    "prob":"heart_disease", 
#    "layers":[5,5], 
#    "activations":["relu","relu"], 
#    "lr":0.5, 
#    "batch_size":10, 
#    "epochs":10
#    "train": 80
# }
"""
@app.route("/api/model/run", methods=["POST"])
def model():
    if request.method == 'POST':
        try:
            # Reformat Data from request to Dataframe
            dataset = dataset_util.initialize_from_json(request.json["data"])

            # Create DataHandler with Dataframe
            datahandler = DataHandler(dataset)

            # Create Model Handler Object from Request
            modelhandler = ModelHandler(**request.json)

            # Create Model with Defined Charcateristics
            modelhandler.clear()
            modelhandler.createModel()

            # Set inputs and outputs
            datahandler.setInputs()
            datahandler.setOutput()

            # Split Dataset Into Training and Test Data
            datahandler.dataset_split(request.json["train"])
            training_features = datahandler.x_train
            training_output = datahandler.y_train

            # Train the Model Using Training Data
            training_result = modelhandler.train(
                training_features, training_output)

            # Evauluate the Model Using Test Data
            test_features = datahandler.x_test
            test_output = datahandler.y_test
            eval_result = modelhandler.evaluate(test_features, test_output)

            return jsonify({
                "msg": "Model Animation was successful",
                "evaluation": eval_result,
                "training": training_result,

            }), 200
        except Exception as e:
            return jsonify({"msg": "An Internal Error Has Occured"}), 500
    return jsonify({"msg": "Method not Allowed"}), 405

#Example: request
# {
#    "data": Dataset
#    "prob":"HRT",
#    "layers":[5,5],
#    "activations":["relu","relu"],
#    "lr":0.5,
#    "batch_size":10,
#    "epochs":10
#    "train": 80
# }


@app.route("/api/model/predict", methods=["POST"])
def predict():
    global modelhandler
    if request.method == 'POST':
        if type(modelhandler) == ModelHandler:
            if request.json["problem"] == "HRT":
                dataframe = pd.read_csv(
                    r"./datasets/Heart Disease -  Binary Classification/Exploration - Heart Disease.csv")
                hndlr = DataHandler(dataframe)
                hndlr = prepHRT(hndlr)
                features = hndlr.x_test
            result = modelhandler.predict(features)
            return jsonify({"msg": "Prediction was Successful", "dataset": result}), 200
        else:
            return jsonify({"msg": "The Training Model have not been initialized"}), 500
    return jsonify({"msg": "Method not Allowed"}), 405
