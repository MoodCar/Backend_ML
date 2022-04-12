from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def main_page():
    return "This is main page."

if __name__ == '__main__':
    app.run(host = '3.34.209.23',port=5000)
