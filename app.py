from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def main_page():
    return "This is main page."

if __name__ == '__main__':
    app.run(host='0.0.0.0')
