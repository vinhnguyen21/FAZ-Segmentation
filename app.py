from flask import Flask, request, redirect, json
import flask_cors
from flask_cors import CORS
import os
from src.classifier import FAZ_Classifier


application = Flask(__name__)
CORS(application)
# IMAGE_FOLDER = os.path.abspath('images')
SOURCE = os.path.abspath("images")
# SOURCE = "/Users/macbook5/Desktop/aws/app/static/images"
IMAGE_FOLDER = os.path.join(SOURCE, "raw")
PREDICTED_FOLDER = os.path.join(SOURCE, "predict")
classifier = FAZ_Classifier()

##### checking error in server ######

@application.route("/")
def index():
    return "API"
    
#http error
@application.errorhandler(404)
def url_error(e):
    return application.response_class(
        response="""  Wrong url
         <pre>{}</pre>""".format(e),
         status= 404,
         mimetype= 'html/text'
    )

# checking internal error
@application.errorhandler(500)
def url_error(e):
    return application.response_class(
        response=""" An internal error occured: 
        <pre>{}</pre>""".format(e),
        status= 500,
        mimetype= "html/text"
    )

@application.route("/faz/predict", methods = ["GET"])
def predict():
    imageID = request.args.get('id')
    image_path = os.path.join(IMAGE_FOLDER, imageID)
    
    result = classifier.predict(image_path, PREDICTED_FOLDER)


    return application.response_class(
        response= json.dumps({"image_path": result}),
        status= 200,
        mimetype= 'application/json'
    )


if __name__ == "__main__":
    application.run(host= '0.0.0.0', port=2000)