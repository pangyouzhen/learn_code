# curl -X GET 'http://127.0.0.1:8082/reader?username=test'

from flask import Flask,request


app = Flask(__name__)

@app.route("/reader",methods=["GET"])
def reader():
    username = request.args.get("username")
    return username

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8082,debug=True)