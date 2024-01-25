# 使用官方的 Python 3 镜像作为基础镜像
FROM python:3

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . /app

# 安装 Flask 及其依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 Flask 应用运行的端口
EXPOSE 5000

# 定义默认的启动命令
CMD ["python", "app.py"]
