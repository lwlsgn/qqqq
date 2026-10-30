#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install roboflow')

from roboflow import Roboflow
rf = Roboflow(api_key="YnpMrUPzqvCFRBQ1q51g")
project = rf.workspace("-irjog").project("box-form-crrection")
version = project.version(1)
dataset = version.download("yolov8")


# In[2]:


### Roboflow에 저장된 이미지 데이터셋을 불러와, YOLOv8 인공지능 모델을 학습시킬 준비를 하는 과정 ###

#Roboflow 패키지를 설치
#!pip install roboflow


#Roboflow의 기능(클래스)을 현재 코드에서 사용할 수 있도록 불러
#from roboflow import Roboflow

#Roboflow 서버에 접근하기 위한 출입증을 제시
#rf = Roboflow(api_key="    ")

#"tc-s"라는 이름의 워크스페이스(팀 또는 개인 작업 공간)로 찾아감
#"mousepen"(마우스와 펜을 구별하거나 인식하는 프로젝트)이라는 이름의 특정 프로젝트를 선택
#project = rf.workspace("tc-s").project("mousepen")

#프로젝트의 2번째 버전 데이터셋을 선택
#version = project.version(2)

#데이터셋의 저장 형식을 YOLOv8 모델이 바로 읽고 학습할 수 있는 구조로 맞춰서 다운로드
#dataset = version.download("yolov8")



# In[3]:


### 다운로드한 데이터셋을 YOLOv8 모델을 학습(Train)  ###

# os, yaml: 파일 경로를 다루고 설정 파일(data.yaml)을 수정하기 위한 기본 라이브러리
import os
import yaml

#윈도우(Windows)나 맥(Mac) 환경에서 머신러닝 라이브러리가 충돌하여 프로그램이 강제 종료되는 에러(OpenMP 에러)를 방지하기 위한 안전장치
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#OLOv8 모델을 사용하기 위한 핵심 모듈 불러옴
from ultralytics import YOLO



## 학습 환경 및 경로 설정 ##
def main():
    #yolov8n.pt: 사전 학습된 모델을 불러옴
    model = YOLO("yolov8n.pt")

    #데이터 경로 자동 수정: 코드가 실행되는 현재 폴더(current_dir)를 기준으로 data.yaml 파일의 내용을 수정
    current_dir = os.getcwd()
    yaml_path = os.path.join(current_dir, "data.yaml")

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        yaml_data['path'] = current_dir
        yaml_data['train'] = 'train/images'  #학습용(train) 데이터와 검증용(val) 데이터를 모두 'train/images'
        yaml_data['val'] = 'train/images'

        #지정한 경로(yaml_path)에 한글이 깨지지 않는 파일(utf-8)을 쓰기 모드(w)로 열고
        #파이썬 데이터(yaml_data)를 안전한 yaml 형식(safe_dump)으로 기록
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(yaml_data, f)


    ## 모델 학습(train)  ##
    model.model.args['verbose'] = False   #화면에 출력되는 복잡한 학습 로그들을 최소화하여 깔끔하게 보여줌

    model.train(
        data=yaml_path,
        epochs=30,    #데이터를 30번 반복 학습
        imgsz=320,    #이미지 크기를 320x320 픽셀
        batch=4,      #한 번에 4장의 이미지만 묶어 처리 (메모리가 부족한 환경에 적합)
        workers=0,    #데이터를 불러올 때 다중 처리(멀티프로세싱)를 사용하지 않음. 윈도우 환경에서 발생하는 에러를 막기 위한 필수 설정
        device='cpu', #래픽카드(GPU)를 사용하지 않고 오직 CPU만으로 학습을 진행
        val=False,    #매 에포크마다 진행하는 성능 평가(검증) 과정을 생략. 학습 속도를 극대화하기 위한 설정
        name='train', #학습 결과가 저장되는 폴더(runs/detect/train)가 이미 존재해도 오류를 내지 않고 덮어쓰거나 이어서 씀
        exist_ok=True
    )

if __name__ == "__main__":
    main()


# In[1]:


### best.pt 파일을 사용하여 웹캠으로 실시간 객체 탐지  ###

import os
import cv2
from ultralytics import YOLO

#clear_output: 주피터 노트북 환경에서 화면에 글씨가 무한정 길어지지 않게 이전 글씨를 지워주는 기능
from IPython.display import clear_output

#머신러닝 라이브러리 충돌로 인한 프로그램 강제 종료를 막아주는 안전장치
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # best.pt 파일의 경로를 자동으로 찾아 추론용 모델로 불러옴. 이 모델은 "객체(예:마우스, 펜)을 구별할 줄 아는 상태
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "runs", "detect", "train", "weights", "best.pt")

    print(f"🎯 가중치 로드: {model_path}")
    model = YOLO(model_path)

    #웹캠 연
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 장치를 인식할 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()  #cap.read(): 카메라에서 찰나의 순간(1프레임)을 사진처럼 찰칵 찍어서 frame에 저장
        if not ret:
            break

        clear_output(wait=True)
        print("🎥 YOLOv8 실시간 탐지 작동 중... (종료: 영상 창 클릭 후 q)")

        # 실시간 추론 (verbose=False로 하단 텍스트 로그 완벽 차단)
        # 찍은 사진(frame)을 모델에 넣어 분석
        results = model(frame, stream=True, verbose=False)

        annotated_frame = frame
        for result in results:
            annotated_frame = result.plot()  #네모 박스(바운딩 박스)와 물체 이름(라벨) 구
            summary = result.verbose()
            names = result.names

            print(summary)  #화면 아래 텍스트 창에 "현재 화면에 마우스 1개, 펜 2개가 보입니다" 같은 요약 텍스트를 출력
            print(names)    #모델이 알고 있는 전체 사물 이름의 목록(딕셔너리)을 출력

        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame) #1장의 사진을 화면

        # cv2.waitKey(1): 팝업창이 떠 있는 상태에서 키보드 입력을 0.001초 기다림
        # 'q'를 누르면, 화면 멈춤
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #카메라 끄기, 띄워졌던 영상 팝업 닫
    cap.release()
    cv2.destroyAllWindows()
    clear_output(wait=False)
    print("👋 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()

