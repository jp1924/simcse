# JP_ver SimCSE
**`주의사항`**

 1. requirements.txt시 pytorch는 맨 마지막에 설치
 
	>     pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

	transformers나 sentence-transformers를 설치 할 때 pytorch도 같이 깔리게 됨. 			
	이때 설치되는 순서에 따라 버전이 모듈이 정상적으로 작동하지 않을 수 있음. 
	때문에 pytorch는 맨 마지막에 설치하는 걸 권장.

 2. output_dir에는 오직 경로만

	 학습된 결과는 [unsup]SimCSE라는 폴더에

	> [mon-day-hour:min:sec]SimCSE-model_name

	와 같은 이름으로 저장된다. 
	경로명을 입력 시 
	
	> training_args.output_dir + f"/{DIR_NAME}"

	와 같은 경로명이 됨
 3. skt/kobert는 지원 안함
	 skt/kobert는 gluonnlp의 sentencepiece tokenizer를 사용하기 때문에 모델을 로딩하는 도중에 에러가 발생함
	 skt/kobert대신 monologg kobert를 사용하는 걸 추천 


	>     https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L194-L212

	>     https://github.com/jeongukjae/KR-BERT-SimCSE/blob/main/model.py#L106-L122

	>     https://github.com/BM-K/KoSimCSE-SKT/blob/main/model/loss.py#L21-L29
