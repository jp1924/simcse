# simcse
주의사항
1. train.py폴더의 OurTrainingArguments에서 output_dir의 올바른 경로를 설정해 줘야함.
  1-1 저장하는 폴더 내에 같은 이름의 폴더가 있으면 에러가 발생함. 그래서 time모듈을 이용해 이름을 달라지게 만들었으니 
  이름부분 말고 경로 부분을 손대는 것을 추천함
  
torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
