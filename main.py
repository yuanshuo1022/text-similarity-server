from flask import Flask, request, abort
from flask_cors import *

from controller.TextSimController import similarity_route
from controller.TrainController import train_route

app = Flask(__name__)
CORS(app)


# # 定义允许访问的IP白名单
# allowed_ips = ['127.0.0.1', '192.168.1.4']  # '127.0.0.1', '192.168.1.1'
#
#
# # 在每个请求之前执行的拦截器

# 在每个请求之前执行的拦截器
@app.before_request
def before_request():
    if request.method == 'OPTIONS':
        # 处理 OPTIONS 请求并返回正确的 CORS 头信息
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers[
            'Access-Control-Allow-Headers'] = 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,X-Web-Server-Auth'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length,Content-Range'
        return response
    print("\n验证：", request.headers.get('X-Web-Server-Auth'))
    if request.headers.get('X-Web-Server-Auth') != 'yuanshuo1022':
        print("\n请求IP:", request.remote_addr)
        abort(403)


app.register_blueprint(train_route)
app.register_blueprint(similarity_route)
# 载入Word2Vec 模型
if __name__ == '__main__':
    app.run(port=5000)
