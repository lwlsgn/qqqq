import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# 1. 페이지 레이아웃 및 스타일 설정
st.set_page_config(
    page_title="Boxing AI Form Trainer",
    page_icon="🥊",
    layout="wide"
)

# 2. 모델 로드 (캐싱 적용)
@st.cache_resource
def load_yolo_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    elif os.path.exists("yolov8n.pt"):
        return YOLO("yolov8n.pt")
    return None

model = load_yolo_model()

if model is None:
    st.error("저장소에서 모델 파일('best.pt')을 찾을 수 없습니다.")
    st.stop()

# 3. 세션 기록(History) 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 4. 자세별 피드백 및 체크포인트 정의
POSE_CRITERIA = {
    "upper": {
        "title": "오른손 어퍼컷 (Right Uppercut)",
        "guide_img": "guide_upper.jpg",
        "checkpoints": [
            "오른팔 팔꿈치가 몸통(옆구리)에 90도(ㄴ자)로 밀착되었는가",
            "주먹의 타격점이 자신의 턱끝(입술) 높이를 초과하지 않고 수직으로 멈췄는가",
            "왼손이 턱과 왼쪽 볼을 감싸는 기본 가드를 유지하고 있는가",
            "오른발 뒤꿈치를 들고 골반 회전으로 추진력을 실었는가"
        ],
        "feedback_high": "체크포인트에 완벽히 부합하는 정석 궤적입니다. 팔꿈치 각도와 타격 높이가 콤팩트합니다.",
        "feedback_mid": "동작은 인식되었으나 주먹이 턱 위로 오버슈팅되거나 팔꿈치가 벌어졌을 위험이 있습니다. 90도 각도를 조이세요.",
        "feedback_low": "어퍼컷 형태가 불안정합니다. 하체 회전과 가드 손의 위치를 점검하세요."
    },
    "jab": {
        "title": "잽 (Left Jab)",
        "guide_img": "guide_jab.jpg",
        "checkpoints": [
            "왼팔이 목표 지점으로 직선 궤적으로 뻗었는가",
            "타격 순간 오른손 가드가 턱 옆에 견고히 붙어있는가",
            "어깨가 턱을 보호하도록 충분히 솟아올라 있는가"
        ],
        "feedback_high": "직선 궤적과 가드 유지가 우수한 정석 잽입니다.",
        "feedback_mid": "오른손 가드가 내려가거나 궤적이 바깥으로 샜을 수 있습니다.",
        "feedback_low": "동작 판별이 불분명합니다. 팔을 직선으로 쭉 뻗어주세요."
    }
}

# 5. 추론 및 결과 분석 함수
def analyze_frame(img):
    results = model(img)
    annotated_img = None
    action_name = "미감지"
    confidence = 0.0

    for r in results:
        # 시각화 이미지 변환 (BGR -> RGB)
        im_array = r.plot()
        annotated_img = Image.fromarray(im_array[..., ::-1])

        # 감지 정보 추출 (Bounding Box or Classification)
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            top_box = r.boxes[0]
            cls_id = int(top_box.cls[0].item())
            action_name = r.names[cls_id]
            confidence = float(top_box.conf[0].item())
        elif hasattr(r, 'probs') and r.probs is not None:
            top_cls = int(r.probs.top1)
            action_name = r.names[top_cls]
            confidence = float(r.probs.top1conf)

    return annotated_img, action_name, confidence

# 6. 결과 출력 컴포넌트 (가이드 1:1 비교, 점수 게이지, 코칭 팁)
def render_analysis_dashboard(orig_img, proc_img, action, conf):
    pose_info = POSE_CRITERIA.get(action.lower(), {
        "title": f"기타 동작 ({action})",
        "guide_img": "guide_default.jpg",
        "checkpoints": ["기본 자세 정렬 및 가드 유지"],
        "feedback_high": "동작이 명확하게 감지되었습니다.",
        "feedback_mid": "동작의 디테일을 다듬을 필요가 있습니다.",
        "feedback_low": "식별 신뢰도가 낮습니다."
    })

    score_pct = int(conf * 100)

    # 1) 상단 점수 카드 및 판정 게이지 (기능 2 & 3)
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("감지된 자세", pose_info["title"])
    col_metric2.metric("정확도 점수", f"{score_pct}점")
    
    if score_pct >= 80:
        col_metric3.success("등급: S (교과서 정자세)")
    elif score_pct >= 60:
        col_metric3.warning("등급: B (교정 필요)")
    else:
        col_metric3.error("등급: C (동작 불안정)")

    st.write(f"**판정 신뢰도 (Confidence):** {score_pct}%")
    st.progress(score_pct / 100.0)

    st.divider()

    # 2) 1:1 비교 뷰어 (기능 1)
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📍 내 분석 결과 (AI 판별)")
        st.image(proc_img if proc_img else orig_img, use_container_width=True)

    with col_right:
        st.subheader("🎯 정석 가이드 레퍼런스")
        guide_path = pose_info["guide_img"]
        if os.path.exists(guide_path):
            st.image(guide_path, caption=f"정석 {pose_info['title']} 기준 폼", use_container_width=True)
        else:
            st.info(f"💡 저장소에 `{guide_path}` 사진을 추가하면 이곳에 정석 사진이 자동으로 뜹니다.")

        st.markdown("**📋 필수 체크포인트**")
        for cp in pose_info["checkpoints"]:
            st.write(f"- {cp}")

    # 3) 세부 코칭 코멘트 (기능 2)
    st.markdown("### 💬 AI 코칭 피드백")
    if score_pct >= 80:
        st.success(pose_info["feedback_high"])
    elif score_pct >= 60:
        st.warning(pose_info["feedback_mid"])
    else:
        st.error(pose_info["feedback_low"])

    # 4) 이력 저장 (기능 3)
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "score": f"{score_pct}%",
        "status": "통과" if score_pct >= 80 else "교정권고"
    })

# --- 메인 UI 영역 ---
st.title("🥊 복싱 자세 분석 및 피드백 시스템")
st.caption("YOLOv8 기반 어퍼컷 및 주요 타격 자세 정밀 교정 플랫폼")

main_tab1, main_tab2, main_tab3 = st.tabs([
    "🎯 단일 자세 정밀 분석", 
    "🔄 연속 시퀀스 판정 모드", 
    "📊 판정 세션 기록"
])

# TAB 1: 단일 분석 (웹캠 / 단일 사진)
with main_tab1:
    input_mode = st.radio("입력 방식 선택", ["📷 브라우저 웹캠 촬영", "📁 사진 업로드"], horizontal=True)
    target_img = None

    if input_mode == "📷 브라우저 웹캠 촬영":
        cam_file = st.camera_input("카메라 앞에서 자세를 취하고 셔터를 누르세요")
        if cam_file:
            target_img = Image.open(cam_file)
    else:
        up_file = st.file_uploader("분석할 사진을 선택하세요", type=["jpg", "jpeg", "png"], key="single_up")
        if up_file:
            target_img = Image.open(up_file)

    if target_img:
        with st.spinner("자세 벡터 및 관절 궤적 분석 중..."):
            proc_img, action, conf = analyze_frame(target_img)
            render_analysis_dashboard(target_img, proc_img, action, conf)

# TAB 2: 연속 판정 모드 (기능 4)
with main_tab2:
    st.subheader("연속 동작 일괄 분석")
    st.write("타격 준비부터 임팩트, 회수까지의 연속 사진들을 한꺼번에 올려 단계별 변화를 점검합니다.")
    
    multi_files = st.file_uploader(
        "여러 장의 사진을 선택하세요 (Ctrl 또는 Shift 클릭)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if multi_files:
        st.write(f"총 **{len(multi_files)}장**의 프레임이 입력되었습니다.")
        cols = st.columns(min(len(multi_files), 4))
        
        for idx, file in enumerate(multi_files):
            col = cols[idx % 4]
            img = Image.open(file)
            p_img, act, cf = analyze_frame(img)
            
            with col:
                st.image(p_img if p_img else img, caption=f"Frame {idx+1}: {act} ({int(cf*100)}%)", use_container_width=True)
                if cf >= 0.8:
                    st.caption("✅ 정석 궤적")
                else:
                    st.caption("⚠️ 보완 필요")

# TAB 3: 세션 기록 관리 (기능 3)
with main_tab3:
    st.subheader("📈 실시간 판정 이력 로그")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.write(f"**[{item['time']}]** 동작: `{item['action']}` | 점수: `{item['score']}` | 상태: `{item['status']}`")
        
        if st.button("기록 초기화"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("아직 측정된 자세 기록이 없습니다.")
