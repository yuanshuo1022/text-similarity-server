from flask import Flask
from flask_cors import *

from controller.TextSimController import similarity_route
from controller.TrainController import train_route



app = Flask(__name__)
CORS(app, supports_credentials=True)


app.register_blueprint(train_route)
app.register_blueprint(similarity_route)
# 载入Word2Vec 模型
if __name__ == '__main__':
    app.run(port=5000)