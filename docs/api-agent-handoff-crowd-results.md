# API 스키마 확장 요청 — `POST /crowd/results`

분석 서버(Trem Detection AI)가 이미 보내는 필드를 백엔드가 거절하고 있습니다. **분석은 성공하는데 callback이 422라서 결과가 저장되지 않고, 백엔드는 TIMEOUT으로 표시됩니다.**

대상 레포: `visit_servant_api`  
엔드포인트: `POST /crowd/results`  
검증: tsoa `CrowdResultBody` / `CrowdDetectedObject`, `additionalProperties: false` (`throw-on-extras`)

분석 서버를 API에 맞추면 추적·진입/이탈 정보가 사라집니다. **API가 아래 필드를 받도록 스키마를 열어 주세요.**

---

## 원인

현재 분석 callback에 API 타입에 없는 필드가 있습니다. tsoa가 여분 필드를 422로 거절합니다. 객체 0개여도 최상위 `alertEventCount` 때문에 전부 실패합니다.

운영 증상:

- 분석 서버: `analysis done` 후 `callback failed ... HTTP Error 422`
- 백엔드: `analyzed=PENDING` → `TIMEOUT`
- 422를 5회 재시도하는 동안 프레임 파일이 사라져 `failed to read frame`도 같이 납니다

---

## 요청

1. `CrowdResultBody`에 `alertEventCount` 추가
2. `CrowdDetectedObject`에 추적·전환 필드 5개 추가
3. tsoa 라우트 재생성 (`src/routes.ts`가 실제 검증 소스)
4. 여분 필드는 422가 아니라 무시하거나, 아래 필드는 required가 아닌 optional로 받기

기존 필드(`countedPeople`, `objects`, `alert`, `zoneIndex` 등)는 유지합니다.

---

## 추가할 필드

### `CrowdResultBody`

| 필드 | 타입 | 필수 | 의미 |
| --- | --- | --- | --- |
| `alertEventCount` | `number` (정수, 0 이상) | 아니오 | `alertNotify=true`인 객체 수. 이번 프레임에서 알림 전환이 난 개수 |

`alertObjectCount`와 다릅니다.

- `alertObjectCount`: 지금 `alert=true`인 객체 수 (머무르는 중 포함)
- `alertEventCount`: `ENTER` / `ESCALATE` / `EXIT`처럼 **상태가 바뀐** 객체 수

글래스·대시보드에서 “새로 울릴지”는 `alertEventCount` 또는 객체 `alertNotify`를 보면 됩니다. 매 프레임 `alert=true`를 다시 알림으로 쓰면 중복 경보가 납니다.

### `CrowdDetectedObject`

| 필드 | 타입 | 필수 | 의미 |
| --- | --- | --- | --- |
| `trackId` | `number` (양의 정수) | 아니오 | 같은 카메라 스트림에서 프레임 간 동일 객체 ID |
| `previousZoneLevel` | `'danger' \| 'warning' \| 'safe' \| null` | 아니오 | 직전 프레임 구역. 첫 등장이면 `null` |
| `zoneTransition` | `string` | 아니오 | 구역 이동. 아래 enum |
| `alertEvent` | `'ENTER' \| 'STAY' \| 'ESCALATE' \| 'EXIT' \| 'NONE'` | 아니오 | 알림 상태 전환 |
| `alertNotify` | `boolean` | 아니오 | `alertEvent`가 `ENTER` \| `ESCALATE` \| `EXIT`이면 `true` |

`trackId`는 요청 `id`가 아니라 **객체 추적 ID**입니다. 카메라별로 유지됩니다.

### `zoneTransition` 값

| 값 | 의미 |
| --- | --- |
| `NONE` | 구역 변화 없음. 또는 `safe`로 처음 등장 |
| `ENTER_WARNING` | 추적 시작, 현재 `warning` |
| `ENTER_DANGER` | 추적 시작, 현재 `danger` |
| `{FROM}_TO_{TO}` | 예: `WARNING_TO_DANGER`, `DANGER_TO_SAFE`, `SAFE_TO_WARNING`. FROM/TO는 `DANGER` \| `WARNING` \| `SAFE` |

### `alertEvent` 값

| 값 | 언제 | `alertNotify` |
| --- | --- | --- |
| `ENTER` | 알림 꺼짐 → 켜짐 | `true` |
| `ESCALATE` | 알림 유지 + `WARNING_TO_DANGER` | `true` |
| `STAY` | 알림 유지 (같은 단계) | `false` |
| `EXIT` | 알림 켜짐 → 꺼짐 | `true` |
| `NONE` | 알림 없음 유지 | `false` |

알림 최종 신호는 기존처럼 `alert`입니다. `alertNotify`는 **이번 프레임에 사용자 알림을 새로 보낼지**입니다.

---

## 타입 초안

```typescript
export type AnalysisAlertEvent = 'ENTER' | 'STAY' | 'ESCALATE' | 'EXIT' | 'NONE';

export interface CrowdDetectedObject {
  class: string;
  confidence: number;
  bbox: number[];
  zoneLevel?: AnalysisZoneLevel;
  zoneName?: string;
  zoneIndex?: number;
  tramZone?: number;
  tramZoneCandidates?: number[];
  tramDirection?: number;
  zoneGap?: number;
  zoneGapThreshold?: number;
  alert?: boolean;
  alertReason?: AnalysisAlertReason;
  // 추가
  trackId?: number;
  previousZoneLevel?: AnalysisZoneLevel | null;
  zoneTransition?: string;
  alertEvent?: AnalysisAlertEvent;
  alertNotify?: boolean;
}

export interface CrowdResultBody {
  id: number;
  status?: AnalysisStatus;
  countedPeople?: number;
  detectedObjectCount?: number;
  objects?: CrowdDetectedObject[];
  alertLevel?: AnalysisAlertLevel | null;
  alertObjectCount?: number;
  detectedFaceCount?: number;
  faces?: CrowdDetectedFace[];
  raw?: Record<string, unknown>;
  errorMessage?: string;
  // 추가
  alertEventCount?: number;
}
```

`previousZoneLevel`은 JSON `null`이 올 수 있습니다. optional omit과 `null`을 둘 다 받아 주세요.

---

## 예시 callback (분석 서버가 이미 보내는 형태)

```json
{
  "id": 10429804,
  "status": "COMPLETED",
  "countedPeople": 29,
  "detectedObjectCount": 5,
  "objects": [
    {
      "class": "person",
      "confidence": 0.91,
      "bbox": [120, 80, 260, 430],
      "zoneLevel": "danger",
      "zoneName": "C01-danger",
      "zoneIndex": 1,
      "tramZone": 1,
      "tramZoneCandidates": [1],
      "tramDirection": 0,
      "zoneGap": 0,
      "zoneGapThreshold": 2,
      "alert": true,
      "alertReason": "TRAM_ZONE_PROXIMITY",
      "trackId": 7,
      "previousZoneLevel": "warning",
      "zoneTransition": "WARNING_TO_DANGER",
      "alertEvent": "ESCALATE",
      "alertNotify": true
    },
    {
      "class": "person",
      "confidence": 0.84,
      "bbox": [310, 210, 510, 460],
      "zoneLevel": "warning",
      "zoneName": "C01-warning",
      "zoneIndex": 1,
      "alert": true,
      "alertReason": "TRAM_ZONE_PROXIMITY",
      "trackId": 7,
      "previousZoneLevel": "warning",
      "zoneTransition": "NONE",
      "alertEvent": "STAY",
      "alertNotify": false
    }
  ],
  "detectedFaceCount": 4,
  "faces": [{ "confidence": 0.94, "bbox": [120, 80, 260, 230] }],
  "alertLevel": "danger",
  "alertObjectCount": 2,
  "alertEventCount": 1,
  "raw": {}
}
```

`raw`에도 동일 객체 배열이 들어갑니다. 지금 422는 **body / objects의 여분 필드** 때문입니다. `raw`는 `Record<string, unknown>`이라 통과합니다.

실패 callback은 기존과 같습니다. `alertEventCount` 없음.

```json
{
  "id": 10429795,
  "status": "FAILED",
  "errorMessage": "failed to read frame: /upload/visit_servant/analyzed/39_7/....jpg",
  "raw": { "error": "...", "frame_abs_path": "...", "congestionSensorDeviceId": 7 }
}
```

---

## 백엔드 처리 제안

```text
if status == FAILED: 실패 기록
elif objects 중 alertNotify == true: 그 항목만 글래스/대시보드 신규 알림
elif alertObjectCount > 0: 상태 유지 (이미 울리는 중). 매 프레임 재알림하지 않음
else: 결과만 저장
```

지금은 `alert == true`만 보면 4FPS마다 같은 경보가 반복됩니다. `alertNotify`를 쓰는 쪽이 맞습니다.

`id`는 계속 idempotency key입니다.

---

## 확인

- [ ] `POST /crowd/results`가 위 예시 JSON을 200으로 받음
- [ ] 객체 0개 + `alertEventCount: 0`도 200
- [ ] `previousZoneLevel: null` 200
- [ ] 기존 최소 payload (`id`, `status`, `countedPeople`만)도 계속 200
- [ ] tsoa `routes.ts` 재생성됨 (타입만 고치고 라우트 안 고치면 계속 422)

분석 서버는 필드 이름을 이 문서 기준으로 고정합니다. 이름을 바꾸면 알려 주세요.
