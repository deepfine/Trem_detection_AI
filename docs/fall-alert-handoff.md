# 박스 기반 쓰러짐 의심 판정 및 백엔드 인수인계

기준일: 2026-10-16. 추가 자세 모델 없이 기존 사람 검출 박스와 추적 ID를 사용한다. 실제 낙상을 확정하는 모델이 아니라 장기체류와 박스 형태에 따른 쓰러짐 의심 규칙이다.

## 변경 범위

| 구분 | 반영 내용 |
| --- | --- |
| 사람·동물·이동수단 | 사람은 `person`, 동물은 `animal`, 킥보드는 `scooter`. 자전거·오토바이 포함, 차량 제외. 휠체어는 보조 검출 유지 |
| 구역 상황 | 트램 위치와 별도로 `alertEvent`에 진입·체류·격상·이탈 상태 전달 |
| 쓰러짐 의심 | 같은 위치에 10초 이상 체류하고 현재 박스의 가로/세로 비율이 1.2 이상일 때 활성화 |
| 독립 경보 | 쓰러짐 의심은 트램 거리 및 구역과 관계없이 `alert=true` |
| 전송 | `fallState`, `fallEvent`, `fallNotify`, 사유 `PERSON_FALL_SUSPECTED` 추가 |
| 오류 수정 | 활성 의심 중 박스가 잠깐 잘렸다가 같은 위치의 가로 박스로 돌아오면 잘못 `EXIT`를 보내던 오류 수정 |

분석 소스는 [frame_falls.py](../frame_falls.py), 추적·경보 결합은 [frame_objects.py](../frame_objects.py), 전송 필드 필터는 [crowd_analysis_protocol.py](../crowd_analysis_protocol.py)이다. 전체 API 계약은 [BACKEND_INTEGRATION.md](../BACKEND_INTEGRATION.md)를 참조한다.

## 판정 기준

- 기준값: `FALL_DWELL_SECONDS=10.0`, `FALL_BOX_RATIO=1.2`.
- 같은 위치의 기준은 사람 박스 하단 중앙이다. 체류 시작 시 박스 높이의 절반, 최소 10px를 허용 반경으로 사용한다.
- 허용 반경을 벗어나거나 검출 간격이 1초를 초과하면 새 의심을 위한 체류 시간을 다시 누적한다. 잘림·무효 박스도 새 의심을 만들지 않는다.
- 유효한 박스로 장기체류 기준을 충족한 뒤 가로 박스가 되면, 가로 박스가 된 시점부터 10초를 다시 기다리지 않는다.
- 활성 의심은 가로 비율을 벗어나거나 위치가 이동하면 해제한다. 잘린 박스 또는 미관측만으로 회복을 확정하지 않으며, 같은 위치의 가로 박스가 다시 관측되면 유지한다.
- 300초는 동일 ID와 구역 `STAY` 유지의 검증 구간이다. 쓰러짐 의심을 보내기 위해 300초를 기다리는 설정은 아니다.
- 체류 시간은 현재 분석 프로세스의 단조 시계 기준이다. 적체된 과거 프레임을 빠르게 일괄 처리한 결과를 실제 촬영 시간 기준 체류 판정으로 해석하지 않는다.

추적 ID가 종료된 뒤 같은 위치에 새 사람이 나타나면 새 체류 시간을 누적한다. `trackId`는 영구 식별자가 아니며 재시작 시 초기화된다. 백엔드는 요청에 `congestionSensorDeviceId` 또는 `camera.id`를 전달하고, 상태 키는 카메라 ID와 추적 ID를 함께 사용한다.

## 전송 경로

```text
백엔드가 프레임 분석 요청
  → 검출·트램 정책·객체 추적·박스 체류 판정
  → HTTP POST callback
  → 수신 API 검증 및 결과 저장
  → analysis_result 방송
  → 새 쓰러짐 의심은 GPS 필터 없이 risk_alarm 저장 후 risk_info 방송
```

통합 서버 기본 주소는 `POST http://api:3535/crowd/results`이며 `VISIT_SERVANT_RESULT_URL`로 지정한다. 객체 전용 서버는 `OBJECT_RESULT_URL`의 `/object/results`로 전송하며 군중 수·얼굴 필드는 생략한다. 수신 환경에서 사용하는 endpoint의 계약을 먼저 확인한다.

헤더는 `Content-Type: application/json`이다. `ANALYSIS_API_KEY`가 설정되면 `X-Analysis-Key`도 보낸다. 문서와 Git에 실제 키를 기록하지 않는다.

다음은 새 쓰러짐 의심이 발생한 경우의 예시다. 수치는 설명용이며 `raw`는 축약하였다.

```json
{
  "id": 41001,
  "status": "COMPLETED",
  "countedPeople": 12,
  "detectedObjectCount": 1,
  "objects": [
    {
      "class": "person",
      "confidence": 0.94,
      "bbox": [100, 200, 240, 260],
      "trackId": 7,
      "zoneLevel": "safe",
      "zoneName": "safe-default",
      "alert": true,
      "alertReason": "PERSON_FALL_SUSPECTED",
      "previousZoneLevel": "safe",
      "zoneTransition": "NONE",
      "alertEvent": "NONE",
      "alertNotify": true,
      "fallState": "FALL_SUSPECTED",
      "fallEvent": "ENTER",
      "fallNotify": true
    }
  ],
  "alertLevel": "danger",
  "alertObjectCount": 1,
  "alertEventCount": 1,
  "detectedFaceCount": 0,
  "faces": [],
  "raw": {
    "congestionSensorDeviceId": 27,
    "frame_path": "27/frame-41001.jpg"
  }
}
```

`id`는 분석 요청 식별자이고 `trackId`는 카메라 내 객체 추적 식별자이다. `bbox`는 원본 이미지 기준 `[x1, y1, x2, y2]`이다. `countedPeople`은 DM-Count 결과이며 `objects`의 사람 수와 동일한 값이 아니다.

### 상태 필드

| 필드 | 값 | 백엔드 처리 기준 |
| --- | --- | --- |
| `alertEvent` | `ENTER`, `STAY`, `ESCALATE`, `EXIT`, `NONE` | 구역 상황. 트램이 멀어 `alert=false`여도 구역 이벤트는 발생 가능 |
| `fallState` | `NORMAL`, `UNKNOWN`, `FALL_SUSPECTED` | 의심 조건 미충족, 근거 부족, 의심 활성 |
| `fallEvent` | `ENTER`, `STAY`, `EXIT`, `NONE` | 쓰러짐 의심 시작·유지·해제·없음 |
| `fallNotify` | boolean | 쓰러짐 `ENTER`, `EXIT`일 때 `true` |
| `alertNotify` | boolean | 구역 통지 조건 또는 쓰러짐 통지 조건 |
| `alert` | boolean | 트램 정책 또는 활성 쓰러짐 의심에 따른 경보 상태 |
| `alertLevel` | `danger`, `warning`, null | 쓰러짐 의심 활성 시 실제 구역이 safe여도 `danger` |
| `alertObjectCount` | 정수 | 현재 `alert=true`인 객체 수 |
| `alertEventCount` | 정수 | 이번 결과의 `alertNotify=true` 객체 수. 같은 객체의 구역·쓰러짐 이벤트는 중복 집계하지 않음 |

`zoneLevel`은 실제 구역을 보존한다. 쓰러짐 경보를 표시하기 위해 `safe`를 `danger`로 변경하지 않는다. `UNKNOWN`을 쓰러짐 부재 또는 회복 확정으로 해석하지 않는다. 현재 추적기가 검출하지 못한 객체에 대해서는 별도 `EXIT` 객체를 합성해 보내지 않는다.

### 수신 및 알림 처리

수신 API에는 다음 반영이 필요하다.

1. `CrowdDetectedObject`에 세 쓰러짐 필드를 optional로 추가하고, `AnalysisAlertReason`에 `PERSON_FALL_SUSPECTED`를 추가한다.
2. 결과 파서가 필드를 보존하도록 수정한다. tsoa의 추가 필드 거절 설정을 유지할 경우 `npx tsoa spec-and-routes`로 검증 라우트와 Swagger를 재생성한다.
3. 라이브 성공 결과에서 `class=person`, `fallState=FALL_SUSPECTED`, `fallEvent=ENTER`, `fallNotify=true`, `alert=true`를 모두 만족하면 독립 위험 알림을 저장·방송한다. 트램 GPS 필터를 적용하지 않는다.
4. `fallEvent=STAY`는 상태만 갱신하고 새 알림을 반복하지 않는다. `EXIT`는 `analysis_result`로 해제 상태를 전달하며 새 `risk_info`를 생성하지 않는다.
5. replay는 `analysis_result`만 방송하고 쓰러짐 알림 저장·재방송을 수행하지 않는다.

이번 백엔드 구현은 `risk_alarm`의 기존 필드와 분석 결과 JSON을 사용하며 DB 마이그레이션을 추가하지 않는다. 쓰러짐 상세 상태는 분석 결과 JSON에 보존하고 `risk_info`에 포함한다. `risk_alarm`에 쓰러짐 전용 컬럼을 새로 저장하는 변경은 포함하지 않는다.

callback 전송은 요청당 10초 제한, 최대 5회 시도이다. 네트워크 오류·5xx·408은 재시도하고 나머지 4xx는 중단한다. 이 10초는 체류시간과 별개이다. 분석 실패는 `status=FAILED`, `errorMessage`, `raw`로 전달한다. 통합 분석이 부분 실패한 경우에도 성공한 구성요소만 `COMPLETED`로 전달될 수 있으므로 `raw.method`와 오류 로그를 함께 확인한다.

## 저장소 전달 위치

| 저장소 | 브랜치 | 전달 내용 |
| --- | --- | --- |
| [분석 서버](https://github.com/deepfine/Trem_detection_AI/tree/feature/tram-alert-default) | `feature/tram-alert-default` | 판정·전송 코드, Dockerfile, 테스트, 본 문서와 검증 요약 |
| [수신 API](https://github.com/deepfine/visit_servant_api/tree/feature/bbox-fall-alert) | `feature/bbox-fall-alert` | 쓰러짐 필드 수신·파싱·독립 위험 방송·관련 테스트·재생성 API 계약 |

분석 기능 커밋은 `034bba9`이다. 수신 API는 기존 작업 폴더의 다른 변경을 제외하고 기준 커밋 `eb42870`에서 분리하였다. 수신 API의 [인수인계 문서](https://github.com/deepfine/visit_servant_api/blob/feature/bbox-fall-alert/docs/fall-alert-handoff.md)를 함께 확인한다.

## 배포 및 확인 순서

1. 수신 API의 타입·파서·독립 알림 소스와 tsoa 생성물을 먼저 배포한다. 수신 API가 새 필드를 거절하는 상태에서 분석 서버를 먼저 배포하면 callback 422가 발생할 수 있다.
2. 분석 소스와 Docker 이미지를 갱신한다. `frame_falls.py`가 이미지에 포함되어야 한다. 기존 검출·군중·얼굴 가중치는 유지하며 새 자세 가중치는 필요하지 않다.
3. 분석 프로세스를 재시작한다. 추적 ID와 체류 누적값은 초기화된다.
4. 분석 서버 `GET /health`에서 다음 응답을 확인한다.

```json
{
  "ok": true,
  "model_loaded": true,
  "fall_detection_enabled": true,
  "fall_detection_method": "bbox_dwell"
}
```

5. 분석 로그의 `analysis posted ... status=COMPLETED`와 API 수신 성공을 확인한다. 운영 DB에 임의 쓰러짐 이벤트를 넣는 시험은 이번 검증에 포함하지 않았다.
6. 화면에는 `FALL_SUSPECTED`를 쓰러짐 의심으로 표시한다. 확정 낙상으로 표기하지 않는다.

## 검증 결과 및 산출물

| 항목 | 결과 | 검증 범위 |
| --- | --- | --- |
| 분석 저장소 테스트 | 82개 통과 | 시간·박스 비율 경계, 잘림·무효 좌표, 위치 이동, ID별 상태 분리·종료, callback 등 |
| API 관련 테스트 | 33개 통과 | 수신 계약·파서·독립 방송·replay·알림 저장 로직. 분리 브랜치에서도 재확인 |
| 실제 영상 재생 | 77.24초, 310샘플 | 원본 시간 기준 4FPS 오프라인 재생. 실제 운영 속도 시험이 아님 |
| 출력 일치 | 영상 310프레임 및 판정 1,357행 확인 | 동일 판정 데이터를 영상과 JSONL에 기록 |
| 쓰러짐 이벤트 | 0건 | 검토된 쓰러짐 정답이 없어 정확도 미산출 |
| 전체 백엔드 타입 검사 | 기준·분리 브랜치 모두 기존 오류 28건 | 신규 오류 0건. 전체 빌드 통과를 의미하지 않음 |

원본 `test_video.mkv`의 SHA-256은 `ed81c1b090e8c183759d1740fa5ba14ac43bb316365d62d15cbf6bf60314e034`이다. 검증 시점 기준이며 향후 입력이 바뀌면 다시 확인한다.

검증 메타데이터는 [fall-validation-summary.json](fall-validation-summary.json)에 보관한다. 재생 스크립트는 [replay-fall-validation.py](replay-fall-validation.py)이다. 서버의 상세 산출물은 `/home/deepfine/trem_detection/result/fall-validation-20261016-di54XT/`에 있다.

- `annotated.mp4`: 판정 확인 영상
- `objects.jsonl`: 프레임·추적 ID별 실제 출력
- `frames.csv`: 프레임 처리 내역
- `tests.xml`: 분석 테스트 결과
- `summary.json`: 검증 요약

영상과 가중치는 저장소에서 제외한다. 재생 스크립트 실행에는 기존 모델 파일, `test_video.mkv`, 서비스와 같은 Python 의존성 및 GPU가 필요하다. 저장소 루트에서 실행하면 `result` 아래 새 폴더를 만들며 운영 callback과 DB 쓰기는 수행하지 않는다.

```bash
python docs/replay-fall-validation.py
```

실제 쓰러짐·눕기·숙이기·박스 병합에 대한 정확도, ID 연속성의 정답 대비 정확도, 전체 실시간 지연·적체·드롭은 미검증이다. 상태 계산과 계약 테스트 통과를 실제 낙상 검출 성능으로 해석하지 않는다.
