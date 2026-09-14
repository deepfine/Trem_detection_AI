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
  "targetClasses": ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "cart"],
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
| `targetClasses` | 선택 | 생략 시 서버 기본 검출 클래스 사용 |
| `tramZone` | 접근 판정 시 필수 | 현재 트램 Zone 번호 |
| `tramDirection` | 선택 | Zone 번호 증가 방향 `1`, 감소 방향 `-1`, 방향 미적용 `0` |
| `tramZoneUncertainty` | 선택 | 트램 Zone 후보 범위, 기본값 `0` |
| `tramObservedAt` | 선택 | timezone을 포함한 ISO-8601 위치 관측 시각 |
| `tramPositionMaxAgeSeconds` | 선택 | 관측 위치의 최대 유효 시간 |
| `forwardZoneGapThreshold` | 선택 | 진행 방향 허용 Zone 간격 |
| `rearZoneGapThreshold` | 선택 | 후방 허용 Zone 간격 |
| `failSafeOnMissingTramPosition` | 선택 | `tramZone` 누락 시 사람 알림 유지 여부, 기본값 `false` |

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
8. 이전 알림 상태와 현재 알림 상태를 비교하여 `alertEvent`와 `alertNotify`를 산출한다.
9. AI 서버가 결과 callback을 전송한다.

자전거, 오토바이, 킥보드, 휠체어와 리어카는 검출·저장 대상이지만 현재 알림 대상은 사람으로 한정한다. 위험 POLYGON에 객체가 있어도 트램 접근 조건을 충족하지 않으면 `alert=false`이다. 백엔드는 `zoneLevel`만으로 알림을 생성하지 않고 `alert`를 최종 신호로 사용한다.

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

`alert`는 현재 프레임의 경보 상태이므로 객체가 위험 범위에 머무르는 동안 `true`를 유지한다. `alertNotify`는 `ENTER`, `ESCALATE`, `EXIT` 시점에만 `true`이므로 신규 알림의 반복 전송을 억제한다. `WARNING_TO_DANGER` 전이 중 기존 알림이 유지되는 경우 `alertEvent=ESCALATE`를 산출한다.

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
