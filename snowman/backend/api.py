from flask_restful import Resource, Api, reqparse
from flask import Flask, request
from snowman.model.text_model import TextModel

app = Flask(__name__)
api = Api(app)

model = TextModel()
model.load()

class ModelApi(Resource):
	def get(self):
		return "nothing here"

	def put(self):
		url = request.form['url']
		score = model.predict(str(url))[0][0]
		return {"input query" : url, 
			"score" : str(score)}

api.add_resource(ModelApi,"/model")

if __name__ == '__main__':
	app.run(port=33507)