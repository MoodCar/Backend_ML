# pip freeze > requirements.txt     // 패키지 목록 생성
# pip install -r requirements.txt     // 패키지 목록 읽어서 설치


"""
  $ python3 app.py: 파이썬 서버 구동시키기

  $ Ctrl + Z: 프로세스 중지하기

  $ bg: 백그라운드에서 서버를 다시 구동시키기

  $ disown -h: 소유권 포기하기

  $ netstat -nap | grep {port_num}

  $ kill -9 {process_num}
"""

from flask import Flask, request, jsonify, abort, make_response
from sentiment.sentiment_model import get_sentiment_model_class
from keyword_extractor.keyword_model import get_keyword_model_class
import yaml

app = Flask(__name__)


with open('./sentiment/configs/cfg.yaml') as f:
    sentiment_cfg = yaml.load(f, Loader=yaml.FullLoader)
with open('./keyword/configs/cfg.yaml') as f:
    keyword_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
sentiment_model_class = get_sentiment_model_class(sentiment_cfg['model_name'])
sentiment_model = sentiment_model_class(sentiment_cfg)

keyword_model_class = get_keyword_model_class(keyword_cfg['model_name'])
keyword_model = keyword_model_class(keyword_cfg)

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
    
    
    
    emotion = sentiment_model.predict(content)
    hashtag = keyword_model.predict(content)

    return jsonify(emotion = emotion, hashtag1 = hashtag[0], hashtag2 = hashtag[1], hashtag3 = hashtag[2])

@app.errorhandler(500)
def internal_server_error(e):
    abort(make_response(jsonify(message = "Internal Server Error"),500))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.register_error_handler(500,internal_server_error)
