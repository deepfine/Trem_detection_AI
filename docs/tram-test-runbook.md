# 트램 테스트 대응 (2026-10-21)

검출 서버는 2026-10-20 14:12부터 `car`·`bus`·`truck`을 검출 결과로 보낸다. 수집 영상에 트램이 없었으므로, 트램이 `bus`나 `truck`으로 검출되는지는 확인하지 못했다. 트램이 그렇게 검출되면, 트램이 CCTV 위험 구역을 지날 때 트램 자체가 위험 알람을 만든다. 이 알람은 GPS 기반 존 접근 알람과 다르다.

## 확인 방법

- 트램 통과 직후 발생한 알람의 객체 `class`가 `bus` 또는 `truck`인지 본다.
- 박스가 그려진 이미지는 `/upload/visit_servant/analyzed/<카메라>/<연>/<월>/<일>/`에 있다. 트램에 `truck - danger`처럼 라벨이 붙어 있으면 트램 오인이다.

## 트램 오인 시 조치

두 조치 모두 버스·트럭 검출만 끈다. 사람, 자전거, 승용차, 동물 검출은 그대로 유지된다.

### 1. 검출 서버에서 끄기 (권장, analysis만 재시작, 약 30초)

`/app/visit_servant_api/compose.yaml`의 analysis `OBJECT_CLASS_CONFIDENCE`에서 `bus`, `truck` 값을 `1.0`으로 바꾼 뒤 재생성한다. 검출 점수는 1.0 미만이므로(현장 트럭 최고 약 0.96) 버스·트럭은 출력되지 않는다.

```text
OBJECT_CLASS_CONFIDENCE: "car=0.45,bus=1.0,truck=1.0,animal=0.45,person=0.65,bicycle=0.5"
```

```bash
cd /app/visit_servant_api
docker compose up -d --no-deps --no-build analysis
docker ps --filter name=visit-servant-analysis   # healthy 확인
```

### 2. 백엔드 요청에서 제외 (백엔드 코드 수정·배포 필요)

백엔드는 지금 `targetClasses`를 보내지 않으므로 검출 서버의 기본 대상을 사용한다. 현재 백엔드 코드에는 이 값을 넣는 부분이 없다. 분석 요청에 다음 값을 넣으면 그 요청부터 적용된다.

```json
"targetClasses": ["person", "bicycle", "motorcycle", "scooter", "wheelchair", "car", "animal"]
```

### 되돌리기

- 조치 1: `bus=0.45,truck=0.45`로 되돌리고 같은 명령으로 재생성한다.
- 조치 2: 요청에서 `targetClasses`를 빼면 원래 상태로 돌아간다.
- 모델 전체를 이전 상태로 되돌릴 때는 `compose.yaml.bak-20261020-ft3`를 복원한다. 이 경우 차량과 개를 다시 거의 검출하지 못한다.

## 테스트 중 수집

검출 쪽에서 10월 21일 20:00까지 선로 카메라 프레임을 GPU1에 수집한다. 운영 GPU0은 사용하지 않는다.

- 트램처럼 큰 `bus`·`truck`·`train` 박스가 나온 프레임
- 전 카메라 원본 2초 간격

테스트가 끝나면 트램이 실제로 어떻게 검출됐는지 정리한다. 오인이 있으면 트램을 `train`으로 라벨하여 재학습한다.
