import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# 1. 페이지 레이아웃 및 커스텀 스타일
st.set_page_config(
    page_title="Boxing AI Form Trainer",
    page_icon="🥊",
    layout="wide"
)

# 기본 "Clear photo" 영문 텍스트 감추기 및 UI 최적화 CSS
st.markdown("""
<style>
    /* 기본 카메라 입력창의 칙칙한 영문 버튼 숨김 보정 */
    button[data-testid="baseButton-secondary"] div:has(p) p {
        font-weight: bold;
    }
    .step-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# 2. 모델 로드 (캐싱)
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

# 3. 세션 상태 관리 (카메라 키 및 연속 프레임 저장소)
if "history" not in st.session_state:
    st.session_state.history = []
if "cam_key_single" not in st.session_state:
    st.session_state.cam_key_single = 0
if "cam_key_seq" not in st.session_state:
    st.session_state.cam_key_seq = 0
if "sequence_frames" not in st.session_state:
    st.session_state.sequence_frames = []

# 4. 판정 기준 및 체크포인트 정의
POSE_CRITERIA = {
    "upper": {
        "title": "오른손 어퍼컷 (Right Uppercut)",
        "guide_img": "guide_upper.jpg",
        "checkpoints": [
            "오른팔 팔꿈치가 옆구리에 90도(ㄴ자)로 견고히 밀착되었는가",
            "주먹 높이가 자신의 턱끝(입술) 높이에서 정확히 멈추었는가",
            "반대편 왼손이 볼과 턱을 감싸는 안면 가드를 유지하는가",
            "오른발 뒤꿈치가 들리며 골반 회전력이 수직으로 실렸는가"
        ],
        "feedback_high": "체크포인트에 완벽히 부합하는 정석 어퍼컷입니다. 콤팩트한 궤적과 가드가 훌륭합니다.",
        "feedback_mid": "동작은 인식되었으나 주먹이 턱 위로 솟구치거나 팔꿈치가 몸통에서 벌어졌을 수 있습니다.",
        "feedback_low": "어퍼컷 형태가 불안정합니다. 하체 회전과 팔꿈치 각도를 점검하세요."
    },
    "jab": {
        "title": "잽 (Left Jab)",
        "guide_img": "guide_jab.jpg",
        "checkpoints": [
            "왼팔이 목표 지점으로 직선 궤적으로 뻗었는가",
            "타격 순간 오른손 가드가 턱 옆통수를 보호하고 있는가"
        ],
        "feedback_high": "직선 궤적과 안면 방어가 우수한 정석 잽입니다.",
        "feedback_mid": "가드가 내려가거나 팔꿈치가 바깥으로 샜을 수 있습니다.",
        "feedback_low": "동작 식별이 불안정합니다. 팔을 직선으로 뻗어주세요."
    }
}

# 5. 추론 함수
def analyze_frame(img):
    results = model(img)
    annotated_img = None
    action_name = "미감지"
    confidence = 0.0

    for r in results:
        im_array = r.plot()
        annotated_img = Image.fromarray(im_array[..., ::-1])

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

# 6. 개별 결과 대시보드 출력
def render_analysis_dashboard(orig_img, proc_img, action, conf):
    pose_info = POSE_CRITERIA.get(action.lower(), {
        "title": f"기타 동작 ({action})",
        "guide_img": "guide_default.jpg",
        "checkpoints": ["기본 자세 정렬 및 가드 유지"],
        "feedback_high": "동작이 명확하게 감지되었습니다.",
        "feedback_mid": "동작 디테일을 다듬을 필요가 있습니다.",
        "feedback_low": "식별 신뢰도가 낮습니다."
    })

    score_pct = int(conf * 100)

    # 상단 요약 지표
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("감지된 자세", pose_info["title"])
    col_metric2.metric("정확도 점수", f"{score_pct}점")
    
    if score_pct >= 80:
        col_metric3.success("등급: S (교과서 정자세)")
    elif score_pct >= 60:
        col_metric3.warning("등급: B (보완 필요)")
    else:
        col_metric3.error("등급: C (동작 불안정)")

    st.write(f"**신뢰도 지수:** {score_pct}%")
    st.progress(score_pct / 100.0)
    st.divider()

    # 1:1 비교 뷰어
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📍 내 자세 판별 결과")
        st.image(proc_img if proc_img else orig_img, use_container_width=True)

    with col_right:
        st.subheader("🎯 정석 가이드 레퍼런스")
        guide_path = pose_info["guide_img"]
        if os.path.exists(guide_path):
            st.image(guide_path, caption=f"정석 {pose_info['title']} 기준 폼", use_container_width=True)
        else:
            st.info(f"💡 저장소에 `{guide_path}` 사진을 추가하면 이곳에 정석 사진이 노출됩니다.")

        st.markdown("**📋 필수 체크포인트**")
        for cp in pose_info["checkpoints"]:
            st.write(f"- {cp}")

    # 코칭 코멘트
    st.markdown("### 💬 AI 맞춤 코칭 피드백")
    if score_pct >= 80:
        st.success(pose_info["feedback_high"])
    elif score_pct >= 60:
        st.warning(pose_info["feedback_mid"])
    else:
        st.error(pose_info["feedback_low"])

    # 히스토리 기록
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "score": f"{score_pct}%",
        "status": "통과" if score_pct >= 80 else "교정권고"
    })

# --- 메인 탭 네비게이션 ---
st.title("🥊 복싱 자세 분석 및 피드백 시스템")
st.caption("YOLOv8 기반 어퍼컷 및 시퀀스 타격 자세 정밀 교정 플랫폼")

tab_single, tab_sequence, tab_history = st.tabs([
    "🎯 단일 자세 정밀 분석", 
    "🔄 연속 시퀀스 판정 모드 (웹캠 지원)", 
    "📊 판정 세션 기록"
])

# ----------------- [TAB 1: 단일 정밀 분석] -----------------
with tab_single:
    input_mode = st.radio("입력 방식 선택", ["📷 브라우저 웹캠 촬영", "📁 사진 업로드"], horizontal=True)
    target_img = None

    if input_mode == "📷 브라우저 웹캠 촬영":
        cam_file = st.camera_input(
            "자세를 취하고 아래 버튼을 눌러 촬영하세요", 
            key=f"cam_single_{st.session_state.cam_key_single}"
        )
        
        # 직관적인 재촬영 제어 버튼
        if cam_file is not None:
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                if st.button("🔄 사진 지우고 다시 찍기", key="btn_reset_single", use_container_width=True):
                    st.session_state.cam_key_single += 1
                    st.rerun()
            target_img = Image.open(cam_file)

    else:
        up_file = st.file_uploader("분석할 사진을 선택하세요", type=["jpg", "jpeg", "png"], key="up_single")
        if up_file:
            target_img = Image.open(up_file)

    if target_img:
        with st.spinner("자세 분석 중..."):
            proc_img, action, conf = analyze_frame(target_img)
            render_analysis_dashboard(target_img, proc_img, action, conf)


# ----------------- [TAB 2: 연속 시퀀스 모드] -----------------
with tab_sequence:
    st.subheader("연속 동작(시퀀스) 실시간 측정 및 분석")
    st.write("준비(Setup) ➔ 타격(Impact) ➔ 복귀(Recovery) 단계를 웹캠으로 차례차례 찍어 연결된 흐름을 평가합니다.")

    seq_mode = st.radio("시퀀스 데이터 입력 방식", ["📷 웹캠 연속 단계별 촬영", "📁 다중 파일 일괄 업로드"], horizontal=True)

    # 1) 웹캠 연속 촬영 워크플로우
    if seq_mode == "📷 웹캠 연속 단계별 촬영":
        current_step_num = len(st.session_state.sequence_frames) + 1
        st.markdown(f"#### 📸 현재 촬영 순서: **Step {current_step_num} 프레임**")

        seq_cam_file = st.camera_input(
            f"Step {current_step_num} 자세를 취하고 셔터를 누르세요",
            key=f"seq_cam_{st.session_state.cam_key_seq}"
        )

        if seq_cam_file is not None:
            temp_img = Image.open(seq_cam_file)
            p_img, act, conf = analyze_frame(temp_img)

            st.write(f"**현재 프레임 판정 결과:** `{act}` ({int(conf*100)}%)")

            # 조작 버튼 3종 (직관적인 한국어 UI)
            btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 2])
            
            with btn_col1:
                if st.button("➕ 이 자세 시퀀스에 등록하고 다음 컷 찍기", type="primary", use_container_width=True):
                    st.session_state.sequence_frames.append({
                        "step": current_step_num,
                        "image": temp_img,
                        "proc_image": p_img,
                        "action": act,
                        "conf": conf
                    })
                    st.session_state.cam_key_seq += 1 # 카메라 자동 리셋
                    st.rerun()

            with btn_col2:
                if st.button("🔄 방금 찍은 컷 버리고 다시 찍기", use_container_width=True):
                    st.session_state.cam_key_seq += 1
                    st.rerun()

            with btn_col3:
                if st.button("🗑️ 시퀀스 전체 초기화", use_container_width=True):
                    st.session_state.sequence_frames = []
                    st.session_state.cam_key_seq += 1
                    st.rerun()

        st.divider()

        # 누적된 시퀀스 프레임 시각화
        if st.session_state.sequence_frames:
            st.markdown(f"### 🎞️ 등록된 연속 시퀀스 타임라인 (총 {len(st.session_state.sequence_frames)}단계)")
            
            frame_cols = st.columns(max(len(st.session_state.sequence_frames), 1))
            total_score = 0
            
            for idx, item in enumerate(st.session_state.sequence_frames):
                col = frame_cols[idx]
                score = int(item["conf"] * 100)
                total_score += score
                
                with col:
                    st.image(item["proc_image"], caption=f"Step {item['step']}: {item['action']}", use_container_width=True)
                    if score >= 80:
                        st.success(f"{score}점 (정석)")
                    else:
                        st.warning(f"{score}점 (보완)")

            avg_score = int(total_score / len(st.session_state.sequence_frames))
            st.markdown(f"#### 🏆 연속 동작 종합 평균 점수: **{avg_score}점**")
            if avg_score >= 80:
                st.balloons()
                st.success("매끄러운 궤적과 일관된 가드를 유지했습니다!")
            else:
                st.info("특정 단계(준비 혹은 타격 회수)에서 팔 각도나 가드가 흐트러졌습니다. 타임라인을 확인해 보세요.")

    # 2) 파일 일괄 업로드 모드
    else:
        multi_files = st.file_uploader(
            "시퀀스 사진들을 여러 장 선택하세요 (Ctrl 또는 Shift 클릭)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )

        if multi_files:
            st.write(f"총 **{len(multi_files)}개**의 시퀀스 프레임 분석 중...")
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


# ----------------- [TAB 3: 세션 기록 관리] -----------------
with tab_history:
    st.subheader("📈 실시간 판정 이력 로그")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.write(f"**[{item['time']}]** 동작: `{item['action']}` | 점수: `{item['score']}` | 상태: `{item['status']}`")
        
        if st.button("기록 초기화", key="btn_clear_history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("아직 측정된 자세 기록이 없습니다.")
