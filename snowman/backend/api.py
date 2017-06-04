from flask_restful import Resource, Api, reqparse
from flask import Flask, request
from keras.models import load_model

app = Flask(__name__)
api = Api(app)

model = load_model("../../fixtures/models/model_0.0.1")

class ModelApi(Resource):
	def get(self):
		return "nothing here"

	def put(self):
		url = request.form['url']
		score = model.predict_proba(url)
		return {"input url" : url, 
			"score" : score}

api.add_resource(ModelApi,"/model")

if __name__ == '__main__':
	app.run()