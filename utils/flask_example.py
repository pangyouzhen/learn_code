import json

from flask import Flask, request
from loguru import logger

#  这里后面可以改为 fastapi
# curl --location --request POST 'http://127.0.0.1:8082/reader' --header 'Content-Type: application/json' --data-raw '{"username":"pangyouzhen"}'
app = Flask(__name__)


@app.route("/reader", methods=["POST"])
def reader():
    data = request.get_data()
    logger.info(data)
    username = json.loads(data)["username"]
    age = json.loads(data)["age"]
    logger.info("username is {}".format(username))
    logger.info("age is {}".format(age))
    return json.dumps({"code": 0, "username": username, "age": age})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082, debug=True)
