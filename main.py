from flask import Flask, request, abort
from flask_cors import *

from controller.TextSimController import similarity_route
from controller.TrainController import train_route

app = Flask(__name__)
CORS(app, supports_credentials=True)

# 定义允许访问的IP白名单
allowed_ips = ['127.0.0.1', '192.168.1.4']  # '127.0.0.1', '192.168.1.1'


# 在每个请求之前执行的拦截器
@app.before_request
def before_request():
    client_ip = request.remote_addr
    print("request.remote_addr: " + request.remote_addr)
    if client_ip not in allowed_ips:
        abort(403)


app.register_blueprint(train_route)
app.register_blueprint(similarity_route)
# 载入Word2Vec 模型
if __name__ == '__main__':
    app.run(port=5000)
