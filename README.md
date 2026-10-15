# Trem Detection AI

CCTV 또는 백엔드 중계 영상을 입력받아 얼굴 비식별화, 객체 탐지, 위험구역 판정, 이동 추적 및 군중 계수를 수행한다. 군중 계수는 DM-Count 밀도추정 결과만 사용한다. YOLO의 `person` 검출 수는 군중 계수에 포함하지 않으며, YOLO는 위험객체 및 위험행동 분석에만 사용한다.

## 기능 범위

| 기능             | 구현   | 모델·방식                      |
| ---------------- | ------ | ------------------------------- |
| 얼굴 블러        | 구현됨 | SCRFD 얼굴 검출 + Gaussian blur |
| 객체 탐지        | 구현됨 | Objects365 YOLO26n + 이동수단 YOLOv8s |
| 객체 추적        | 구현됨 | 프레임 간 중심점 추적           |
| 위험구역 판정    | 구현됨 | 안전·경고·위험 다각형 구역    |
| 접근 위험 판정   | 구현됨 | 객체 Zone·트램 Zone 간격 및 진행 방향 |
| 군중 계수        | 구현됨 | DM-Count UCF-QNRF 밀도추정      |
| 군중 계수 정확도 | 미측정 | 운영 CCTV 정답 라벨 필요        |

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
- OpenCV, NumPy, PyTorch, ONNX Runtime GPU, Ultralytics, InsightFace, Norfair

```bash
pip install opencv-python numpy torch onnxruntime-gpu ultralytics insightface norfair pytest
```

전체 분석에는 다음 모델 파일이 필요하다.

```text
result/models/scrfd_det_10g.onnx
result/models/objects365_yolo26n.onnx
result/models/mobility_yolov8s.onnx
result/models/dm_count_qnrf.pth
```

모델 파일은 용량 때문에 Git 저장 대상에서 제외하였다.

객체 분석의 기본 대상은 사람, 자전거, 오토바이, 킥보드 및 동물이다. 휠체어는 교통약자 유형 판정을 위한 보조 객체로 포함한다.

| 객체 유형 | 세부 객체 | 결과 class | 비고 |
| --- | --- | --- | --- |
| 사람 | 성인 | `person` | 사람 검출 가능 |
| 사람 | 교통약자 | `person` | `person` 유지. 기존 영상 처리 경로에서 이동보조기기 연관 기반 유형 추정 가능 |
| 사람 | 아동 | `person` | `person` 유지. 기존 영상 처리 경로에서 원근·신장 기반 유형 추정 가능 |
| 이동수단 | 자전거 | `bicycle` | 검출 가능 |
| 이동수단 | 오토바이 | `motorcycle` | 검출 가능 |
| 이동수단 | 킥보드 | `scooter` | 이동수단 보조 모델 사용. 탑승자만 `person`으로 검출될 수 있음 |
| 동물 | 고양이, 강아지 등 | `animal` | 모델에서 검출한 동물 종류를 단일 class로 통합 |
| 기타 | 승용차, 버스, 트럭 | `car`, `bus`, `truck` | 분류 판정에서 제외. `targetClasses`로 요청해도 결과에서 제외 |

동물 대상은 `anomaly_rules.py`의 `ANIMAL_CLASSES`에 정의한다. 기본 대상에서 카트·가방·여행가방도 제외한다. `targetClasses`는 기본 대상 중 분석할 클래스를 선택하며, 생략 시 전체 기본 대상을 사용한다. 모델의 동물 종류별 라벨은 모두 `animal`로 통합한다. `targetClasses`에는 `animal`을 지정한다.

사람의 검출 class는 유형 추정과 관계없이 `person`을 유지한다. 아동·성인·이동보조기기 사용자 추정은 `process_blur_anomalies.py`의 기존 `person_group` 출력이며, 서버 객체 응답에는 아직 포함되지 않는다. 추정 근거가 부족하면 `person_uncertain`을 유지한다.

이동수단 모델은 `Bike`, `Pedestrian`, `Scooter`, `Wheelchair` 중 `scooter`, `wheelchair` 검출을 보강한다. 가중치와 Ultralytics 런타임을 서비스에 배포할 때에는 AGPL-3.0 또는 별도 상용 라이선스 조건을 확인한다.

```bash
curl -L --fail -o yolo26n-objv1-150.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-objv1-150.pt
curl -L --fail -o result/models/mobility_yolov8s.pt \
  https://raw.githubusercontent.com/rolson24/BWCT-Tracker/electron-app/backend/tracking/models/yolov8s-2024-02-16-best.pt

python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26n-objv1-150.pt").export(format="onnx", imgsz=640, simplify=True, opset=17)
YOLO("result/models/mobility_yolov8s.pt").export(format="onnx", imgsz=640, simplify=True, opset=17)
PY
mv yolo26n-objv1-150.onnx result/models/objects365_yolo26n.onnx
```

## 쓰러짐 의심 판정

추가 자세 모델 없이 기존 `person` 박스와 추적 ID를 사용한다. 같은 위치에 10초 이상 머물고 현재 박스의 가로/세로 비율이 1.2 이상이면 `FALL_SUSPECTED`로 판정한다. 기준은 `frame_falls.py`의 `FALL_DWELL_SECONDS`, `FALL_BOX_RATIO`이다. 장기체류 기준과 300초 이상 동일 ID 유지 기능은 별개이다.

위치는 박스 하단 중앙을 기준으로 하며 체류 시작 박스 높이의 절반(최소 10px)을 벗어나면 시간을 다시 누적한다. 검출 간격이 1초를 넘거나 박스가 잘리면 체류 시간을 초기화한다. 같은 위치에 계속 머문 상태에서 박스가 가로로 길어지면 다시 10초를 기다리지 않는다. 박스가 정상 비율로 돌아오거나 이동해 조건을 벗어나면 해제한다. 잘린 박스만으로 활성 경보를 해제하지 않는다.

쓰러짐 의심은 트램 거리·구역과 관계없이 `alert=true`, `alertReason=PERSON_FALL_SUSPECTED`를 산출한다. 기존 구역 `alertEvent`와 별도로 `fallEvent`를 전달한다. `/health`에서 `fall_detection_method=bbox_dwell`을 확인한다. 눕기·숙이기·사람 간 박스 병합을 실제 쓰러짐과 구분하는 정확도 검증은 필요하다.

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

| 항목                 |      결과 | 상태   |
| -------------------- | --------: | ------ |
| 평가 프레임          |        78 | 실측   |
| 평균 추정 인원       |   33.90명 | 실측   |
| 최소·최대 추정 인원 | 0명·74명 | 실측   |
| 평균 추론시간        |  40.29 ms | 실측   |
| p95 추론시간         |  45.59 ms | 실측   |
| 정확도·MAE·RMSE    |         - | 미측정 |

위 인원 수는 모델 출력이며 정답 인원과 비교한 정확도가 아니다. UCF-QNRF 기반 모델은 고밀도 군중 사진에 최적화되어 있으므로, 운영 CCTV 표본에 인원 정답을 부여한 뒤 MAE와 RMSE를 산출해야 한다.

## ROI 객체 분석 Docker 실행

백엔드 요청·callback 전체 규격은 [`BACKEND_INTEGRATION.md`](BACKEND_INTEGRATION.md)를 참조한다.

운영 경로는 카메라별 `danger`, `warning`, `safe` POLYGON에서 객체를 검출한다. 객체 바운딩 박스의 하단 중앙점을 객체가 위치한 구역의 기준점으로 사용한다. POLYGON 밖은 `safe-default`로 판정한다. 기존 2선 ROI JSON도 하위 호환으로 지원한다. 프레임은 HTTP로 전송하지 않고 백엔드와 AI 컨테이너가 공유하는 경로에서 읽는다.

| 화면 표시 | JSON `level` | 판정 및 전달 기준 |
| --- | --- | --- |
| 위험 | `danger` | 최우선 구역이며 접근 조건 충족 시 위험 알림 대상 |
| 주의·경계 | `warning` | 접근 조건 충족 시 주의 알림 대상 |
| 안전 | `safe` | 모니터링만 수행하며 알림 없음 |
| 안전(미지정 영역) | `safe-default` | POLYGON 밖 객체에 AI 서버가 자동 부여 |

객체가 여러 POLYGON에 동시에 포함되면 `danger > warning > safe` 순으로 판정한다. 백엔드에는 한글 표시명이 아니라 `zoneLevel`의 영문 enum과 `zoneName`을 전달한다. 현재 구현에서 주의와 경계는 별도 단계가 아니라 `warning` 한 단계이다.

```bash
docker build -t trem-object-ai .

docker run --rm --gpus all \
  --name trem-object-ai \
  -p 8080:8080 \
  -p 8765:8765 \
  --add-host host.docker.internal:host-gateway \
  -e OBJECT_DEVICE_ID=0 \
  -e OBJECT_RESULT_URL=http://host.docker.internal:3535/object/results \
  -v "$(pwd)/result/models/objects365_yolo26n.onnx:/app/result/models/objects365_yolo26n.onnx:ro" \
  -v "$(pwd)/result/models/mobility_yolov8s.onnx:/app/result/models/mobility_yolov8s.onnx:ro" \
  -v "/upload/visit_servant/analyzed:/upload/visit_servant/analyzed:ro" \
  trem-object-ai
```

상태 확인 endpoint는 `GET /health`, 분석 요청 endpoint는 `POST /object/analysis`이다. 분석 요청은 비동기로 접수되며 HTTP 202를 반환한다.

```json
{
  "id": 41,
  "congestionSensorDeviceId": 27,
  "tramZone": 6,
  "tramDirection": 1,
  "tramZoneUncertainty": 1,
  "tramObservedAt": "2026-09-01T10:20:30+09:00",
  "tramPositionMaxAgeSeconds": 2.0,
  "forwardZoneGapThreshold": 3,
  "rearZoneGapThreshold": 1,
  "failSafeOnMissingTramPosition": true,
  "frame_abs_path": "/upload/visit_servant/analyzed/27/frame.jpg",
  "zone": {
    "width": 1280,
    "height": 720,
    "zoneGapThreshold": 2,
    "zones": [
      {
        "name": "track-z8",
        "level": "danger",
        "index": 8,
        "points": [[120, 180], [80, 650], [700, 650], [650, 180]]
      },
      {
        "name": "rail-z8",
        "level": "warning",
        "index": 8,
        "points": [[650, 180], [700, 650], [1200, 650], [1050, 170]]
      }
    ]
  },
  "targetClasses": ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "animal"]
}
```

`width`와 `height`는 좌표 작성 기준 해상도이며 실제 이미지 해상도에 맞춰 자동 보정한다. `tramZone`과 `zones[].index`의 차이로 접근 여부를 판정한다. `tramDirection`은 `-1`, `0`, `1`을 사용하며, 진행 방향과 후방 임계값을 각각 적용할 수 있다. `tramZoneUncertainty`가 있으면 후보 Zone 전체 중 최소 간격을 적용한다. `tramZone`이 누락된 경우 `failSafeOnMissingTramPosition=true`이면 보수적 알림을 생성한다. 관측 시각이 `tramPositionMaxAgeSeconds`를 초과하면 설정값과 관계없이 위치 만료 알림을 생성한다. 현재 `alert=true` 적용 대상은 `danger` 또는 `warning` 구역의 사람으로 한정한다.

완료 결과는 `OBJECT_RESULT_URL`로 전달한다.

```json
{
  "id": 41,
  "status": "COMPLETED",
  "detectedObjectCount": 2,
  "objects": [
    {
      "class": "person",
      "confidence": 0.9123,
      "bbox": [120, 80, 260, 430],
      "zoneLevel": "danger",
      "zoneName": "track-z8",
      "zoneIndex": 8,
      "tramZone": 6,
      "tramZoneCandidates": [5, 6, 7],
      "tramDirection": 1,
      "zoneGap": 1,
      "zoneGapThreshold": 3,
      "alert": true,
      "alertReason": "TRAM_POSITION_UNCERTAIN"
    },
    {
      "class": "scooter",
      "confidence": 0.84,
      "bbox": [310, 210, 510, 460],
      "zoneLevel": "safe",
      "zoneName": "safe-default",
      "alert": false
    }
  ],
  "alertLevel": "danger",
  "alertObjectCount": 1,
  "raw": {
    "method": "objects365_yolo26n+mobility_yolov8s_roi",
    "inference_ms": 17.2,
    "frame_abs_path": "/upload/visit_servant/analyzed/27/frame.jpg",
    "congestionSensorDeviceId": 27
  }
}
```

`objects[].bbox`는 원본 이미지 기준 `[x1, y1, x2, y2]` 좌표이다. 얼굴 좌표는 통합 분석 `POST /crowd/results`의 `faces[]`로 전달하며, AI 서버는 원본 이미지를 수정하지 않는다. 백엔드는 얼굴 좌표를 사용하여 모자이크 또는 블러를 적용한다.

## Visit Servant 연동

`visit_servant_api`가 공유 경로의 JPEG를 `POST /crowd/analysis`로 넘기면, 이 서버가 DM-Count로 인원을 세고 `POST /crowd/results`로 결과를 돌려준다. 이미지 바이트는 HTTP로 받지 않는다.

```bash
python crowd_analysis_server.py \
  --model result/models/dm_count_qnrf.pth \
  --device cuda:0 \
  --analyzed-dir /upload/visit_servant/analyzed \
  --result-url http://127.0.0.1:3535/crowd/results
```

| 변수                         | 기본값                              | 역할                 |
| ---------------------------- | ----------------------------------- | -------------------- |
| `FRAME_ANALYZED_DIR`       | `/upload/visit_servant/analyzed`  | 상대경로를 붙일 루트 |
| `CROWD_MODEL_PATH`         | `result/models/dm_count_qnrf.pth` | DM-Count 가중치      |
| `CROWD_DEVICE`             | `cuda:0`                          | 추론 장치            |
| `VISIT_SERVANT_RESULT_URL` | `http://api:3535/crowd/results`   | 결과 회신 URL        |
| `OBJECT_MODEL_PATH`        | `result/models/objects365_yolo26n.onnx` | 일반 객체 모델 |
| `MOBILITY_MODEL_PATH`      | `result/models/mobility_yolov8s.onnx` | 킥보드·휠체어 보강 모델 |
| `OBJECT_DEVICE_ID`         | `0` | 객체 탐지 GPU ID |
| `ANALYSIS_API_KEY`         | 비움                                | `X-Analysis-Key`   |

요청 본문은 `id`, `frame_abs_path`(또는 `frame_path`), `congestionSensorDeviceId`를 필수 식별 정보로 사용한다. 거리 판정 시 `tramZone`과 선택 정책 필드를 추가한다. 결과는 `countedPeople`, 객체·얼굴 좌표, 구역 판정, 알림 정보 및 `status=COMPLETED|FAILED`를 포함한다.

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

`crowd_analysis_server.py`가 기동되면 구역 편집기도 함께 열린다. 브라우저에서 `http://localhost:8765`에 접속해 카메라 최신 분석 JPEG 위에 `danger`, `warning`, `safe` POLYGON과 객체 Zone 번호를 저장한다. Docker로 띄울 때는 `-p 8765:8765`로 외부에 연다.

파일은 `/upload/visit_servant/zones/{장비ID}.json`이다. 분석 요청마다 `congestionSensorDeviceId`로 파일을 다시 읽는다. 중첩 POLYGON은 `danger > warning > safe` 순으로 판정하고, 지정 POLYGON 밖은 `safe-default`로 분류한다.

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

| 파일                           | 역할                                                 |
| ------------------------------ | ---------------------------------------------------- |
| `object_analysis_server.py`  | POLYGON 구역 객체 검출 및 결과 전달 서버             |
| `crowd_analysis_server.py`   | `visit_servant_api` 핸드오프 HTTP 서버             |
| `crowd_analysis_protocol.py` | 분석 요청 경로 해석 및 결과 JSON                     |
| `crowd_counter.py`           | DM-Count 추론 및 영상 단독 평가                      |
| `process_blur_anomalies.py`  | 얼굴 블러·객체 탐지·위험 판정·군중 계수 통합 처리 |
| `anomaly_rules.py`           | 구역 진입 및 접근 위험 규칙                          |
| `zone_annotator.py`          | 카메라별 위험구역 웹 편집기                          |
| `camera_zones.py`            | 장비 ID ↔ 구역 JSON 매칭                            |
| `zones.example.json`         | 다각형 구역 예시                                     |
