#  Moodcar - Backend_ML
>2022년 1학기 SW캡스톤디자인 - 감정 일기 서비스
<br/>

##  API 문서
[Moodcar - APIs](https://canonn11.gitbook.io/moodcar_ml-apis/)


## 사용법

### sentiment model

<br/>


KoBERT, SBERT로 이루어져 있습니다.

모델 옵션은 yaml을 이용합니다.

    model_name : KoBERT or SBERT

    batch_size: 한 번에 처리할 문장 개수

    model_path : 학습한 모델 파라미터 파일 경로

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
    

    

    



    
    
