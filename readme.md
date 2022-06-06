#  Moodcar - Backend_ML
>2022년 1학기 SW캡스톤디자인 - 감정 일기 서비스
<br/>


## Introduction
바쁜 현대인들은 하루를 돌아보는 시간을 갖는 것이 쉽지 않다. 일기를 쓰면서 하루를 되돌아 볼 수는 있지만, 아날로그 일기의 경우 어떤 날에 무슨 일이 있었고 어떤 감정을 느꼈는지 되돌아보기 위해서는 일기 전체를 읽어봐야 하므로 시간이 많이 소모된다는 단점이 있다. 또한 일기를 작성할 때 자신이 느낀 감정을 정확하게 파악하지 못할 때도 있다.

감정 일기 서비스 Moodcar는 인공지능을 이용해 일기 내용으로 감정 분포 및 주요 활동을 자동으로 추론한다. 그리고 달력에 보기 좋게 기록한다. 더 나아가 사용자가 느낀 감정에 따라 콘텐츠(음악, 상담사 등)를 추천해줌으로써, 사용자가 감정을 다스릴 수 있게 도와준다.
<br/>


## 사용법

### sentiment model

<br/>


KoBERT, SBERT, Ensemble로 이루어져 있습니다.

모델 옵션은 yaml을 이용합니다.

    model_name : KoBERT or SBERT

    batch_size: 한 번에 처리할 문장 개수

    model_path : 학습한 모델 파라미터 파일 경로
    
    model2_path : ensemble 인 경우, KOTE 데이터 셋으로 학습한 모델 파라미터 파일 경로

    max_len : 문장 최대 길이

    label_name : 감정 종류

      -

      -

      ...
      
    num_workers : num_workers
    
    device : device(cpu, cuda:0 ....)
    
    
### Keyword Extractor

<br/>

tfidf, keybert로 구현되어 있습니다.


공통적으로 들어가는 옵션은

    max_keywords : 뽑을 키워드 개수
    
    model_name : KoBERT or TfIdf or SBERT(not use)

<br/>

TfIdf 옵션

    idf_data_path : idf 학습할 문서들
    
    
<br/>


KeyBERT(KoBERT 기반) 옵션

    batch_size: 한 번에 처리할 문장 개수

    model_path : 학습한 모델 파라미터 파일 경로

    max_len : 문장 최대 길이
    
    num_workers : num_workers
    
    device : device(cpu, cuda:0 ....)
    

    

    



    
    
