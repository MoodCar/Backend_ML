# pip freeze > requirements.txt     // 패키지 목록 생성
# pip install -r requirements.txt     // 패키지 목록 읽어서 설치
from flask import Flask, request, jsonify, abort, make_response
from sentiment import model

app = Flask(__name__)

# 서버 작동 여부 확인 Route ('/')
@app.route('/')
def server_check():
    return jsonify(message="Server Linked!")

# Emotion Prediction Route ('/prediction')
@app.route('/prediction', methods = ['POST'])
def prediction():
    # request를 json으로 받아온다.
    req = request.get_json()
    content = req['content']

    # request input validator
    if not content:
        abort(make_response(jsonify(message="Diary Content is empty"), 400))
    elif len(content) > 10000 :
        abort(make_response(jsonify(message="Invalid Length of diary content"), 400))
    
    
    
    emotion = model.predict(content)




    emotion = 'happy'
    return jsonify(emotion= emotion)

@app.errorhandler(500)
def internal_server_error(e):
    abort(make_response(jsonify(message = "Internal Server Error"),500))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.register_error_handler(500,internal_server_error)
