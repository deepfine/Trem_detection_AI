# 백엔드 연동 규격 — 영역 판정 및 알림

## 연동 기준

백엔드는 원본 프레임을 HTTP body로 전송하지 않는다. 공유 저장소에 JPEG를 저장한 뒤 프레임 경로, 카메라 ID, 영역 좌표와 트램 위치를 AI 서버에 전달한다. AI 서버는 요청을 비동기로 접수하고 분석 완료 후 JSON callback을 전송한다.

```text
백엔드                         AI 서버
  │ JPEG를 공유 경로에 저장       │
  │ POST /object/analysis         │
  │ ────────────────────────────> │
  │ 202 Accepted                  │
  │ <──────────────────────────── │
  │                               │ 객체 검출 → POLYGON 판정
  │                               │ → 트램 Zone 간격 판정
  │ POST /object/results          │
  │ <──────────────────────────── │
  │ 결과 저장 및 알림 송신          │
```

`202 Accepted`는 분석 완료가 아니라 작업 접수만 의미한다. 최종 결과는 `OBJECT_RESULT_URL` callback으로 확인한다.

## 영역 좌표 전달

### 권장 방식 — 분석 요청에 `zone` 포함

백엔드가 영역 설정을 관리하는 경우 각 분석 요청의 `zone`에 POLYGON을 포함한다. `POST /object/analysis`는 이 방식을 사용한다.

```json
{
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
}
```

| 필드 | 필수 | 기준 |
| --- | --- | --- |
| `width`, `height` | 필수 | 영역 좌표를 작성한 원본 이미지 해상도 |
| `zoneGapThreshold` | 선택 | 트램 Zone과 객체 Zone의 기본 허용 간격, 기본값 `2` |
| `zones` | 필수 | 영역 배열 |
| `zones[].name` | 권장 | 백엔드와 대시보드에서 사용할 고유 영역명 |
| `zones[].level` | 필수 | `danger`, `warning`, `safe` 중 하나 |
| `zones[].index` | 접근 판정 시 필수 | 트램 진행축 기준 Zone 번호, `0` 이상의 정수 |
| `zones[].points` | 필수 | 최소 3개의 `[x, y]` 좌표로 구성한 POLYGON |

해상도가 720p이면 `width=1280`, `height=720` 좌표를 사용한다. 실제 프레임 크기가 다르면 AI 서버가 좌표를 비례 보정한다.

| 화면 표시 | `level` | 의미 |
| --- | --- | --- |
| 위험 | `danger` | 접근 조건 충족 시 위험 알림 |
| 주의·경계 | `warning` | 접근 조건 충족 시 주의 알림 |
| 안전 | `safe` | 검출 결과만 저장하고 알림 없음 |
| 안전(미지정 영역) | `safe-default` | AI 서버가 POLYGON 밖 객체에 자동 부여 |

객체가 중첩 영역에 포함되면 `danger > warning > safe` 순으로 하나의 영역을 선택한다. 포함 여부는 객체 bounding box의 하단 중앙점으로 판정한다.

### 저장 영역 방식 — 통합 분석 endpoint

`POST /crowd/analysis`는 `congestionSensorDeviceId`에 해당하는 아래 파일을 우선 사용한다.

```text
/upload/visit_servant/zones/{congestionSensorDeviceId}.json
```

파일이 없을 때만 요청 body의 `zone`을 사용한다. 따라서 저장 파일과 요청 `zone`이 동시에 존재하면 저장 파일이 우선한다. 영역 좌표를 백엔드가 단일 관리하려면 `POST /object/analysis`와 요청 body의 `zone`을 사용하는 구성을 권장한다.

## 분석 요청

### Endpoint

```http
POST /object/analysis
Content-Type: application/json
```

### 요청 예시

```json
{
  "id": 41001,
  "congestionSensorDeviceId": 27,
  "frame_abs_path": "/upload/visit_servant/analyzed/27/frame-41001.jpg",
  "tramZone": 6,
  "tramDirection": 1,
  "tramZoneUncertainty": 1,
  "tramObservedAt": "2026-09-01T10:20:30+09:00",
  "tramPositionMaxAgeSeconds": 2.0,
  "forwardZoneGapThreshold": 3,
  "rearZoneGapThreshold": 1,
  "failSafeOnMissingTramPosition": true,
  "targetClasses": ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "animal"],
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
  }
}
```

| 필드 | 필수 | 기준 |
| --- | --- | --- |
| `id` | 필수 | 양의 정수. 요청과 callback을 연결하는 키 |
| `congestionSensorDeviceId` | 권장 | 카메라·센서 식별자 |
| `frame_abs_path` 또는 `frame_path` | 필수 | AI 컨테이너에서도 접근 가능한 JPEG 경로 |
| `zone` | 객체 전용 endpoint에서 필수 | 영역 좌표 JSON. 미전달 시 요청은 접수되지만 객체 결과는 비어 있음 |
| `targetClasses` | 선택 | 생략 시 사람·자전거·오토바이·킥보드·휠체어·동물 사용. 지정 시 기본 대상 중 선택하며 차량·카트·가방은 제외 |
| `tramZone` | 접근 판정 시 필수 | 현재 트램 Zone 번호 |
| `tramDirection` | 선택 | Zone 번호 증가 방향 `1`, 감소 방향 `-1`, 방향 미적용 `0` |
| `tramZoneUncertainty` | 선택 | 트램 Zone 후보 범위, 기본값 `0` |
| `tramObservedAt` | 선택 | timezone을 포함한 ISO-8601 위치 관측 시각 |
| `tramPositionMaxAgeSeconds` | 선택 | 관측 위치의 최대 유효 시간 |
| `forwardZoneGapThreshold` | 선택 | 진행 방향 허용 Zone 간격 |
| `rearZoneGapThreshold` | 선택 | 후방 허용 Zone 간격 |
| `failSafeOnMissingTramPosition` | 선택 | `tramZone` 누락 시 사람 알림 유지 여부, 기본값 `false` |

사람의 class는 성인·아동·교통약자 구분 없이 `person`을 유지한다. 기존 영상 처리 경로의 유형 추정 결과는 서버 응답에 포함되지 않는다. 동물은 종류와 관계없이 `animal`로 반환한다. 킥보드는 보조 모델의 `scooter` 검출을 사용하며, 탑승자만 `person`으로 검출될 수 있다. 세부 대상 목록은 `anomaly_rules.py`의 `ANIMAL_CLASSES` 및 `OBSTACLE_CLASSES`를 기준으로 한다.

정상 요청은 다음 응답을 즉시 반환한다.

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{"ok": true, "id": 41001}
```

요청 JSON, 경로 또는 필드 형식이 잘못되면 `400`, 모델이 준비되지 않았으면 `503`을 반환한다.

## 판정 순서

1. 요청한 객체 클래스를 검출한다.
2. bounding box의 하단 중앙점이 포함된 POLYGON을 확인한다.
3. 중첩 시 `danger > warning > safe` 우선순위를 적용한다.
4. `zoneIndex`와 `tramZone`의 차이를 계산한다.
5. 객체가 `person`이고 `danger` 또는 `warning`이며 Zone 간격 조건을 충족하면 `alert=true`로 설정한다.
6. 동일 카메라의 객체 위치에 대한 Kalman 예측, 전역 거리 매칭 및 외형 특징 재식별을 수행하여 `trackId`를 부여한다.
7. 이전 Zone과 현재 Zone을 비교하여 `zoneTransition`을 산출한다.
8. 이전 Zone과 현재 Zone으로 `alertEvent`와 `alertNotify`를 산출한다. 트램 위치가 없어도 위험·주의 구역 진입은 `ENTER`, 경고에서 위험으로 올라가면 `ESCALATE`, 체류는 `STAY`, 이탈은 `EXIT`다. 구역 단계 하향(이탈, 위험→주의)은 같은 객체에서 3프레임(약 1.5초) 연속 관측될 때만 반영하며, 그동안 `zoneLevel`·`zoneName`은 직전 구역을 유지하고 `STAY`를 보낸다. 구역 경계에 걸친 정지 객체가 매 프레임 `EXIT`·`ENTER`를 반복하던 문제를 막기 위한 것이다. 진입·`ESCALATE`는 지연 없이 즉시 보낸다. 같은 위치에 머문 객체(주차 자전거 등)는 하향이 60초 연속 유지될 때만 반영한다. 정지해 있던 객체가 검출에서 잠시 빠졌다가 60초 안에 같은 위치·같은 class로 다시 검출되면 이전 `trackId`와 구역 상태를 이어받아 `ENTER`를 다시 보내지 않는다. 이동하던 객체는 이어받지 않으므로 같은 자리에 새로 들어온 사람은 새 `trackId`로 `ENTER`를 받는다.
9. AI 서버가 결과 callback을 전송한다.

자전거, 오토바이, 킥보드, 휠체어와 리어카는 검출·저장 대상이지만 `alert=true`는 사람만 대상으로 한다. 트램 위치가 없으면 `alert=false`이고, 위험·주의 구역에 있는 상황은 `zoneLevel`과 `alertEvent`로 전달한다. 트램 접근 알림은 백엔드가 그 상황과 트램 위치를 맞춰 판단한다.

`tramObservedAt`과 `tramPositionMaxAgeSeconds`가 모두 있고 위치가 만료되면 `failSafeOnMissingTramPosition` 설정과 관계없이 `TRAM_POSITION_STALE` 알림을 적용한다. `tramZone`이 누락된 경우에는 `failSafeOnMissingTramPosition=true`일 때만 `TRAM_POSITION_MISSING` 알림을 적용한다.

## 결과 callback

### Endpoint와 header

```http
POST {OBJECT_RESULT_URL}
Content-Type: application/json
X-Analysis-Key: {ANALYSIS_API_KEY}  # 설정한 경우에만 포함
```

기본 주소는 `http://api:3535/object/results`이다. callback 실패 시 0.5초, 1.0초, 1.5초, 2.0초, 2.5초 간격으로 최대 5회 시도한다.

### 알림 발생 결과

```json
{
  "id": 41001,
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
      "alertReason": "TRAM_POSITION_UNCERTAIN",
      "trackId": 1,
      "previousZoneLevel": "warning",
      "zoneTransition": "WARNING_TO_DANGER",
      "alertEvent": "ESCALATE",
      "alertNotify": true
    },
    {
      "class": "scooter",
      "confidence": 0.84,
      "bbox": [310, 210, 510, 460],
      "zoneLevel": "safe",
      "zoneName": "safe-default",
      "alert": false,
      "trackId": 2,
      "previousZoneLevel": null,
      "zoneTransition": "NONE",
      "alertEvent": "NONE",
      "alertNotify": false
    }
  ],
  "alertLevel": "danger",
  "alertObjectCount": 1,
  "alertEventCount": 1,
  "raw": {
    "method": "objects365_yolo26n+mobility_yolov8s_roi",
    "inference_ms": 17.2,
    "frame_abs_path": "/upload/visit_servant/analyzed/27/frame-41001.jpg",
    "congestionSensorDeviceId": 27
  }
}
```

| 필드 | 백엔드 처리 기준 |
| --- | --- |
| `status` | `COMPLETED`이면 결과 저장, `FAILED`이면 실패 처리 |
| `objects[]` | 검출 객체와 영역 판정 결과 저장 |
| `objects[].bbox` | 원본 이미지 기준 `[x1, y1, x2, y2]` |
| `objects[].zoneLevel` | 대시보드 표시 단계 |
| `objects[].alert` | 글래스 앱·대시보드 알림의 최종 boolean 신호 |
| `objects[].alertReason` | 알림 원인 코드 |
| `objects[].trackId` | 동일 카메라의 활성 추적 구간에서 사용하는 객체 식별자 |
| `objects[].previousZoneLevel` | 직전 검출 프레임의 Zone 단계. 최초 검출은 `null` |
| `objects[].zoneTransition` | Zone 단계 전이. 예: `WARNING_TO_DANGER`, `DANGER_TO_SAFE` |
| `objects[].alertEvent` | 알림 상태 전이: `ENTER`, `ESCALATE`, `STAY`, `EXIT`, `NONE` |
| `objects[].alertNotify` | 신규·격상·해제 통지가 필요한 경우 `true`. `STAY`는 `false` |
| `alertLevel` | 알림 객체 중 최상위 단계. 알림이 없으면 `null` |
| `alertObjectCount` | `alert=true`인 객체 수 |
| `alertEventCount` | `alertNotify=true`인 객체 수 |

`alertReason`은 다음 값을 사용한다.

| 값 | 의미 |
| --- | --- |
| `TRAM_ZONE_PROXIMITY` | 확정 트램 Zone과의 간격 조건 충족 |
| `TRAM_POSITION_UNCERTAIN` | 트램 Zone 불확실성 범위를 적용하여 조건 충족 |
| `TRAM_POSITION_MISSING` | 트램 위치 누락에 대한 fail-safe 적용 |
| `TRAM_POSITION_STALE` | 트램 위치 유효 시간 초과에 대한 fail-safe 적용 |

### 알림 없음

분석은 정상 완료되었으나 알림 대상이 없으면 다음 조건을 충족한다.

```json
{
  "id": 41002,
  "status": "COMPLETED",
  "alertLevel": null,
  "alertObjectCount": 0,
  "objects": []
}
```

실제 callback에는 `detectedObjectCount`와 `raw`도 포함된다. `objects=[]`는 정상적인 미검출 결과이며 실패가 아니다.

### 분석 실패

```json
{
  "id": 41003,
  "status": "FAILED",
  "errorMessage": "failed to read frame: /upload/visit_servant/analyzed/27/frame-41003.jpg",
  "raw": {
    "error": "failed to read frame: /upload/visit_servant/analyzed/27/frame-41003.jpg",
    "frame_abs_path": "/upload/visit_servant/analyzed/27/frame-41003.jpg",
    "congestionSensorDeviceId": 27
  }
}
```

백엔드는 `id`를 idempotency key로 사용하여 callback 재시도에 따른 중복 저장을 방지한다.

## 알림 처리 기준

백엔드의 최소 처리 규칙은 다음과 같다.

```text
if status == "FAILED":
    실패 기록 및 재처리 정책 적용
else:
    현재 경보 상태를 갱신
    objects 중 alertNotify == true인 항목을 순회
    alertEvent가 EXIT이면 기존 알림 해제
    나머지는 신규 또는 격상 알림 전송
```

callback에는 `trackId`, `previousZoneLevel`, `zoneTransition`, `alertEvent`, `alertNotify` 및 최상위 `alertEventCount`를 포함한다. 수신 API의 타입과 생성 라우트에서 해당 필드 수용을 확인하였다.

동일 객체가 계속 검출되고 경보 조건을 충족하면 `STAY`를 유지한다. 체류시간에 대한 10초 제한은 없으며 300초 이후에도 유지 가능하다. 4 FPS 기준 1,201회 연속 합성 검출로 300초 구간의 동일 ID와 `STAY` 유지를 검증하였다. 실제 영상에서의 가림·ID 전환에 대한 검증은 별도 수행이 필요하다. 미검출 추적 보관시간과 체류시간은 서로 다른 조건이다.

`alert`는 트램 접근 판정 결과다. 트램 위치가 없으면 `false`로 두고, 위험·주의 구역 상황은 `alertEvent`로 전달한다. `alertNotify`는 `ENTER`, `ESCALATE`, `EXIT` 시점에만 `true`다. `WARNING_TO_DANGER`이면 `alertEvent=ESCALATE`다.

`trackId`는 `congestionSensorDeviceId`별 Norfair multi-object tracking 결과이다. Kalman 위치 예측과 HSV 외형 특징을 함께 적용하여 객체 교차와 일시 가림 이후의 ID 전환을 억제하며, 최대 30개 분석 프레임의 미검출 상태와 최대 60개 분석 프레임의 외형 재식별 후보를 유지한다. 분석 프로세스 재시작 이후에는 식별자가 재할당되므로 영구 DB 식별자로 사용하지 않는다.

## 통합 분석 callback

군중 수와 얼굴 좌표까지 함께 사용할 때에는 `POST /crowd/analysis`와 `VISIT_SERVANT_RESULT_URL`을 사용한다. 결과에는 객체 필드 외에 다음 필드가 추가된다.

```json
{
  "countedPeople": 12,
  "detectedFaceCount": 1,
  "faces": [
    {"confidence": 0.94, "bbox": [120, 80, 260, 230]}
  ]
}
```

`faces[].bbox`는 원본 이미지 좌표이다. AI 서버는 프레임을 수정하지 않으며 백엔드가 해당 좌표로 모자이크 또는 블러를 적용한다.

통합 분석은 군중·객체·얼굴 구성요소 중 객체 또는 얼굴 분석 오류를 로그로 기록하고 나머지 결과를 `COMPLETED`로 callback할 수 있다. `raw.method`에 `objects365_yolo26n+mobility_yolov8s`가 없으면 `objects=[]`을 안전 판정으로 해석하지 않는다. 객체 알림 결과를 필수 데이터로 취급하는 연동은 객체 전용 endpoint를 사용한다.


## YOLO26 검출 설정

정지 자전거의 프레임 간 복구와 선택적 출처 필드는 [자전거 누락 개선 기록](docs/bicycle-miss-upgrade.md)을 참조한다. `ROI_REDETECTION`은 현재 프레임의 모델 재검출이며, `VISUAL_TRACKER`는 이전 모델 검출을 영상 일치로 보완한 추정이다. `VISUAL_TRACKER`의 `confidence`를 현재 프레임의 YOLO 점수로 해석하지 않는다. 현재 백엔드 수신기의 필드 보존 패치를 적용하기 전에는 `OBJECT_VISUAL_RECOVERY=0`을 유지한다. 현재 백엔드는 정의되지 않은 객체 필드를 HTTP 422로 거부하여 프레임 결과 전체를 잃으므로, AI 서버는 `OBJECT_DETECTION_SOURCE_FIELDS=1`일 때만 출처 필드를 전송한다(기본 `0`).

정확도 중심 헤드·고해상도 입력·FP16·선택적 TensorRT 실행 경로와 검증 결과는 [YOLO26 적용 기록](docs/yolo26-adoption.md)을 참조한다. `OBJECT_MODEL_PATH`로 지정한 ONNX의 입력 크기·정밀도를 자동으로 확인하고, 결과 `bbox`는 원본 이미지 좌표로 복원한다. 주 모델 1280과 보조 모델 640을 함께 사용할 수 있다.

`OBJECT_TENSORRT=1`은 TensorRT 라이브러리와 ONNX Runtime provider 검증 후에 설정한다. 현재 기본값은 CUDA 실행이며, TensorRT 초기화가 실패하면 모델 로딩을 실패 처리한다. 이 설정은 실제 TensorRT 처리 속도가 검증되었다는 의미가 아니다.

## 쓰러짐 의심 결과

변경 내역과 배포·검증 절차는 [쓰러짐 인수인계 문서](docs/fall-alert-handoff.md)를 참조한다.

사람 class는 `person`으로 유지하고 다음 선택 필드를 추가한다. 기존 `alertEvent`는 구역 상황, `fallEvent`는 쓰러짐 상황으로 구분한다.

| 필드 | 값과 의미 |
| --- | --- |
| `fallState` | `UNKNOWN`: 근거 부족 또는 확인 대기, `NORMAL`: 유효 박스이며 의심 조건 미충족, `FALL_SUSPECTED`: 쓰러짐 의심 활성 |
| `fallEvent` | `ENTER`: 새 의심, `STAY`: 의심 유지, `EXIT`: 박스 비율·이동 조건에 따른 의심 해제, `NONE`: 의심 없음 |
| `fallNotify` | `ENTER`, `EXIT`에서 `true` |
| `alertReason` | 쓰러짐 의심 활성 시 `PERSON_FALL_SUSPECTED` |

```json
{
  "class": "person",
  "trackId": 7,
  "zoneLevel": "danger",
  "alert": true,
  "alertReason": "PERSON_FALL_SUSPECTED",
  "alertEvent": "STAY",
  "alertNotify": true,
  "fallState": "FALL_SUSPECTED",
  "fallEvent": "ENTER",
  "fallNotify": true
}
```

활성 쓰러짐 의심은 `alertLevel=danger`로 집계한다. `alertNotify`는 구역 또는 쓰러짐의 통지 조건을 합산하므로 구역 변화 없이도 쓰러짐 시작·해제 시 `alertEventCount`에 포함된다. 실제 발생 지점의 `zoneLevel`은 보존한다.

수신 API는 새 필드를 저장하고 `analysis_result`로 전달한다. 라이브 결과에서 `person`의 `FALL_SUSPECTED`·`fallEvent=ENTER`·`fallNotify=true`·`alert=true`이면 위험구역이며 기존 트램 접근 윈도우 조건을 충족한 경우에만 `DANGEROUS` 위험 알림으로 저장한 뒤 `risk_info`를 방송한다. `STAY`와 replay는 새 쓰러짐 알림을 반복하지 않는다. 해제는 `analysis_result`의 `fallEvent=EXIT`로 전달하며, 해제용 `risk_info`를 새로 생성하지 않는다.

판정은 추가 자세 모델 없이 같은 사람의 박스가 같은 위치에 10초 이상 머물고, 현재 가로/세로 비율이 1.2 이상인 경우에 적용한다. 이동하거나 검출 간격이 1초를 넘으면 체류 시간을 다시 누적한다. 화면 경계에 잘린 박스는 새 의심을 만들지 않으며 활성 의심을 단독으로 해제하지 않는다. 실제 낙상·눕기·숙이기·박스 병합의 구분 정확도는 실제 영상 검증 전이며, 검출 실패나 ID 전환에 따른 누락·중복 가능성이 있다.


쓰러짐 위험신호의 최종 조건은 위험구역(`danger`)과 기존 트램 접근 조건의 동시 충족이다. 안전·주의 구역은 제외한다. 분석 callback은 관측 결과이고, 사용자에게 보내는 `risk_info`는 기존 백엔드 접근 윈도우를 재사용한다. 새 접근 계산·전달 필드·유효시간은 추가하지 않는다.
