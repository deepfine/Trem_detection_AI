# Trem Detection AI

CCTV 또는 백엔드 중계 영상을 입력받아 얼굴 비식별화, 객체 탐지, 위험구역 판정, 이동 추적 및 군중 계수를 수행한다. 군중 계수는 DM-Count 밀도추정 결과만 사용한다. YOLO의 `person` 검출 수는 군중 계수에 포함하지 않으며, YOLO는 위험객체 및 위험행동 분석에만 사용한다.

## 기능 범위

| 기능 | 구현 | 모델·방식 |
|---|---|---|
| 얼굴 블러 | 구현됨 | SCRFD 얼굴 검출 + Gaussian blur |
| 객체 탐지 | 구현됨 | YOLO ONNX |
| 객체 추적 | 구현됨 | 프레임 간 중심점 추적 |
| 위험구역 판정 | 구현됨 | 안전·경고·위험 다각형 구역 |
| 접근 위험 판정 | 구현됨 | 이동 속도 및 방향 |
| 군중 계수 | 구현됨 | DM-Count UCF-QNRF 밀도추정 |
| 군중 계수 정확도 | 미측정 | 운영 CCTV 정답 라벨 필요 |

## 처리 구조

```text
백엔드 중계 영상
  └─ 프레임 수신
      ├─ DM-Count 군중 계수
      ├─ SCRFD 얼굴 검출·블러
      └─ YOLO 객체 탐지
          └─ 추적·위험구역·접근 위험 판정
              ├─ 모자이크 결과 영상
              ├─ 군중 계수 로그
              └─ 위험 이벤트 로그
```

카메라별 중계 URL은 `cameras.json`에 저장한다. 실제 주소와 인증정보가 포함되는 파일이므로 Git 저장 대상에서 제외하였다. 형식은 `cameras.example.json`을 참조한다.

## 실행 환경

- Python 3.10 이상
- NVIDIA GPU 및 CUDA
- OpenCV, NumPy, PyTorch, ONNX Runtime GPU, Ultralytics, InsightFace

```bash
pip install opencv-python numpy torch onnxruntime-gpu ultralytics insightface pytest
```

전체 분석에는 다음 모델 파일이 필요하다.

```text
result/models/scrfd_det_10g.onnx
result/models/yolov8n.onnx
result/models/assistive_yolov8s_worldv2.pt
result/models/dm_count_qnrf.pth
```

모델 파일은 용량 때문에 Git 저장 대상에서 제외하였다.

## DM-Count 가중치

참고 구현과 동일한 UCF-QNRF 가중치를 내려받고 SHA-256을 검증한다.

```bash
mkdir -p result/models
curl -L --fail \
  -o result/models/dm_count_qnrf.pth \
  'https://drive.usercontent.google.com/download?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru&export=download&confirm=t'

echo '16ef954a2cef40c66ee664f69273559553b735d5e2c9f90e2444f3c25dd45e05  result/models/dm_count_qnrf.pth' \
  | sha256sum -c -
```

DM-Count 구현과 가중치는 MIT License를 따른다. 고지문은 `licenses/DM-Count-LICENSE`에 포함하였다.

## DM-Count 단독 평가

YOLO 객체 탐지와 관계없이 DM-Count만 실행한다.

```bash
python crowd_counter.py \
  --source test_video.mkv \
  --sample-fps 1 \
  --device cuda:0 \
  --out-dir result/crowd_evaluation
```

산출물은 다음과 같다.

- `frame_counts.csv`: 프레임 번호, 영상 시각, 추정 인원, 추론시간
- `summary.txt`: 표본 수, 평균·최소·최대 추정 인원, 평균·p95 추론시간

### 현재 평가 결과

2026-08-19 로컬 `test_video.mkv`를 1 FPS로 샘플링하였다.

| 항목 | 결과 | 상태 |
|---|---:|---|
| 평가 프레임 | 78 | 실측 |
| 평균 추정 인원 | 33.90명 | 실측 |
| 최소·최대 추정 인원 | 0명·74명 | 실측 |
| 평균 추론시간 | 40.29 ms | 실측 |
| p95 추론시간 | 45.59 ms | 실측 |
| 정확도·MAE·RMSE | - | 미측정 |

위 인원 수는 모델 출력이며 정답 인원과 비교한 정확도가 아니다. UCF-QNRF 기반 모델은 고밀도 군중 사진에 최적화되어 있으므로, 운영 CCTV 표본에 인원 정답을 부여한 뒤 MAE와 RMSE를 산출해야 한다.

## ROI 객체 분석 Docker 실행

현재 단순화된 운영 경로는 2개 선으로 구성한 ROI 내부의 객체만 검출한다. 객체 바운딩 박스의 하단 중앙점을 ROI 포함 여부의 기준점으로 사용한다. 프레임은 HTTP로 전송하지 않고 백엔드와 AI 컨테이너가 공유하는 경로에서 읽는다.

```bash
docker build -t trem-object-ai .

docker run --rm --gpus all \
  --name trem-object-ai \
  -p 8080:8080 \
  --add-host host.docker.internal:host-gateway \
  -e OBJECT_DEVICE_ID=0 \
  -e OBJECT_RESULT_URL=http://host.docker.internal:3535/object/results \
  -v "$(pwd)/result/models/yolov8n.onnx:/app/result/models/yolov8n.onnx:ro" \
  -v "/upload/visit_servant/analyzed:/upload/visit_servant/analyzed:ro" \
  trem-object-ai
```

상태 확인 endpoint는 `GET /health`, 분석 요청 endpoint는 `POST /object/analysis`이다. 분석 요청은 비동기로 접수되며 HTTP 202를 반환한다.

```json
{
  "id": 41,
  "congestionSensorDeviceId": 27,
  "frame_abs_path": "/upload/visit_servant/analyzed/27/frame.jpg",
  "zone": {
    "width": 1280,
    "height": 720,
    "lines": [
      [[120, 180], [80, 650]],
      [[1050, 170], [1200, 650]]
    ]
  },
  "targetClasses": ["person", "bicycle", "motorcycle", "car", "truck", "bus"]
}
```

`width`와 `height`는 좌표를 작성한 기준 해상도이다. 실제 이미지 해상도가 다르면 좌표를 자동으로 보정한다. `targetClasses`를 생략하면 `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`, `backpack`, `suitcase`를 검출한다.

완료 결과는 `OBJECT_RESULT_URL`로 전달한다.

```json
{
  "id": 41,
  "status": "COMPLETED",
  "detectedObjectCount": 2,
  "objects": [
    {"class": "person", "confidence": 0.9123, "bbox": [120, 80, 260, 430]},
    {"class": "bicycle", "confidence": 0.84, "bbox": [310, 210, 510, 460]}
  ],
  "detectedFaceCount": 1,
  "faces": [
    {"confidence": 0.94, "bbox": [120, 80, 260, 230]}
  ],
  "raw": {
    "method": "dm_count+yolov8n+scrfd",
    "inference_ms": 54.4,
    "face_ms": 5.9
  }
}
```

`faces[].bbox`는 원본 이미지 기준 `[x1, y1, x2, y2]` 좌표이다. AI 서버는 얼굴 영역을 수정하지 않으며, 백엔드는 이 좌표를 사용하여 모자이크 또는 블러 처리를 수행한다.

## Visit Servant 연동

`visit_servant_api`가 공유 경로의 JPEG를 `POST /crowd/analysis`로 넘기면, 이 서버가 DM-Count로 인원을 세고 `POST /crowd/results`로 결과를 돌려준다. 이미지 바이트는 HTTP로 받지 않는다.

```bash
python crowd_analysis_server.py \
  --model result/models/dm_count_qnrf.pth \
  --device cuda:0 \
  --analyzed-dir /upload/visit_servant/analyzed \
  --result-url http://127.0.0.1:3535/crowd/results
```

| 변수 | 기본값 | 역할 |
|---|---|---|
| `FRAME_ANALYZED_DIR` | `/upload/visit_servant/analyzed` | 상대경로를 붙일 루트 |
| `CROWD_MODEL_PATH` | `result/models/dm_count_qnrf.pth` | DM-Count 가중치 |
| `CROWD_DEVICE` | `cuda:0` | 추론 장치 |
| `VISIT_SERVANT_RESULT_URL` | `http://api:3535/crowd/results` | 결과 회신 URL |
| `ANALYSIS_API_KEY` | 비움 | `X-Analysis-Key` |

요청 본문은 `id`, `frame_abs_path`(또는 `frame_path`), `congestionSensorDeviceId`를 사용한다. 결과는 `countedPeople`과 `status=COMPLETED|FAILED`이다.

## 통합 분석

4 FPS로 얼굴 블러·객체 탐지·위험 판정을 수행하고, DM-Count는 4개 분석 프레임마다 실행한다. 이 설정에서 군중 수 갱신 주기는 1초이다.

```bash
python process_blur_anomalies.py \
  --source test_video.mkv \
  --zone zones.json \
  --analysis-fps 4 \
  --crowd-interval 4 \
  --confirm-frames 1 \
  --out-video result/analyzed.mp4 \
  --out-dir result/analyzed
```

주요 산출물은 다음과 같다.

- `analyzed.mp4`: 얼굴 블러, 위험 표시 및 `CROWD N` 오버레이
- `frame_times.csv`: DM-Count 추정 인원과 단계별 처리시간
- `events.csv`: 객체·위험행동 이벤트
- `summary.txt`: 처리량, 군중 계수 평균 및 단계별 성능

## 카메라별 구역 설정

`crowd_analysis_server.py`가 기동되면 구역 편집기도 함께 열린다. 브라우저에서 `http://localhost:8765`에 접속해 카메라 최신 분석 JPEG 위에 점을 찍어 위험구역 다각형을 저장한다.

파일은 `/upload/visit_servant/zones/{장비ID}.json`이다. 분석 요청마다 `congestionSensorDeviceId`로 이 파일을 다시 읽어, 다각형 안의 객체만 결과에 남긴다.

```bash
python zone_annotator.py \
  --analyzed-dir /upload/visit_servant/analyzed \
  --zones-dir /upload/visit_servant/zones \
  --host 0.0.0.0 \
  --port 8765
```

## 테스트

```bash
pytest -q
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| `object_analysis_server.py` | 2개 선 사이의 객체 검출 및 결과 전달 서버 |
| `crowd_analysis_server.py` | `visit_servant_api` 핸드오프 HTTP 서버 |
| `crowd_analysis_protocol.py` | 분석 요청 경로 해석 및 결과 JSON |
| `crowd_counter.py` | DM-Count 추론 및 영상 단독 평가 |
| `process_blur_anomalies.py` | 얼굴 블러·객체 탐지·위험 판정·군중 계수 통합 처리 |
| `anomaly_rules.py` | 구역 진입 및 접근 위험 규칙 |
| `zone_annotator.py` | 카메라별 위험구역 웹 편집기 |
| `camera_zones.py` | 장비 ID ↔ 구역 JSON 매칭 |
| `zones.example.json` | 다각형 구역 예시 |
