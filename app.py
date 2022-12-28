from flask import Flask, request
from main import main

app = Flask(__name__)


@app.route('api/machine-learning/start', methods=['POST'])
def hello_world():
    json = request.get_json()
    result = main(csv=json['csv'], column_base_class=json['column_base_class'],
         stand_scaler=json['stand_scaler'], one_hot_encoder=json['one_hot_encoder'], test_size=json['test_size'],
         algorithm_name=json['algorithm_name'], predict_values=json['predict_values'])

    return result


if __name__ == '__main__':
    app.run()
