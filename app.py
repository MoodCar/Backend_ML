# pip freeze > requirements.txt     // 패키지 목록 생성
# pip install -r requirements.txt     // 패키지 목록 읽어서 설치
from flask import Flask, request, jsonify

app = Flask(__name__)

# 서버 작동 여부 확인 Route ('/')
@app.route('/')
def main_page():
    return "Server Linked!"

# Emotion Prediction Route ('/prediction')
@app.route('/prediction', methods = ['POST'])
def prediction():
    # request를 json으로 받아온다.
    req = request.get_json()
    content = req['content']
    
    # 한 문장씩 파싱 ( 필요시 사용 )
    text=content.replace('\n',' ').replace('  ',' ')
    text=text.split('.')
    #print(text)
    
    


    # return example. ( 추후 수정 )
    emotion = 'happy'
    return jsonify(emotion= emotion)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
