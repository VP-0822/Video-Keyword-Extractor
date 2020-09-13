from flask import Flask, request
from flask_restful import Resource, Api
import pipeline as pp

app = Flask(__name__)
api = Api(app)

class KeywordExtractorAPI(Resource):
    def get(self):
        videoId = request.args.get('videoId')
        predicted_dict = pp.predictForSingleVideo(videoId)
        pp.populateKeywords(predicted_dict)
        return predicted_dict

api.add_resource(KeywordExtractorAPI, '/keywords')

if __name__ == '__main__':
    app.run(debug=True)