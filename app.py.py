import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("🥊 복싱 자세 판별 AI")

# 1. 학습 완료된 모델 불러오기
@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    elif os.path.exists("yolov8n.pt"):
        return YOLO("yolov8n.pt")
    return None

model = load_model()

if model is None:
    st.error("깃허브 저장소에 'best.pt' 파일이 없습니다. 가중치 파일을 업로드해 주세요.")
    st.stop()

# 2. 이미지 입력 (사진 업로드 또는 웹캠 촬영)
st.write("자세 사진을 업로드하거나 웹캠으로 직접 찍어보세요.")
tab1, tab2 = st.tabs(["📁 사진 파일 올리기", "📷 웹캠으로 촬영"])

input_image = None

with tab1:
    uploaded = st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded)

with tab2:
    camera_shot = st.camera_input("자세를 취하고 셔터를 누르세요")
    if camera_shot:
        input_image = Image.open(camera_shot)

# 3. 모델 분석 결과 화면 출력
if input_image is not None:
    st.image(input_image, caption="입력된 이미지", use_container_width=True)
    
    with st.spinner("자세 분석 중..."):
        results = model(input_image)
        
        for r in results:
            # 바운딩 박스가 그려진 이미지 추출
            im_array = r.plot()
            im_rgb = Image.fromarray(im_array[..., ::-1])
            st.image(im_rgb, caption="AI 판별 결과", use_container_width=True)
            
            # 감지된 라벨(upper, jab 등) 확인
            labels = []
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                labels = [r.names[int(b.cls[0].item())] for b in r.boxes]
            elif hasattr(r, 'probs') and r.probs is not None:
                labels = [r.names[int(r.probs.top1)]]
            
            if labels:
                st.success(f"판별 결과: {', '.join(set(labels))}")
            else:
                st.warning("동작이 명확하게 감지되지 않았습니다.")
