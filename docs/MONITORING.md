# FraudX Monitoring

FraudX now includes lightweight production-style prediction monitoring without adding a heavyweight monitoring service.

## What is monitored?

The API records every prediction in MongoDB with:

- fraud probability
- binary prediction
- decision threshold
- model version
- timestamp

The monitoring endpoint compares the current prediction-score distribution with the previous window.

## API

```http
GET /monitoring/predictions?window_hours=24
```

Example:

```json
{
  "window_hours": 24,
  "current": {
    "count": 120,
    "avg_probability": 0.19,
    "fraud_rate": 0.08
  },
  "previous": {
    "count": 115,
    "avg_probability": 0.16,
    "fraud_rate": 0.06
  },
  "prediction_drift": {
    "psi": 0.13,
    "level": "warning"
  },
  "status": "warning"
}
```

## PSI thresholds

| PSI | Status | Interpretation |
|---:|---|---|
| `< 0.10` | stable | Little evidence of distribution shift |
| `0.10–0.25` | warning | Investigate the change |
| `>= 0.25` | drift | Strong distribution shift; review model/data |

These are operational heuristics, not universal statistical guarantees.

## Important limitation

The current monitor detects **prediction-score drift**, not full feature-level drift. Feature drift requires a versioned reference dataset or feature baseline. That is intentionally kept separate from the online prediction monitor so the project does not pretend to have a reference distribution it does not actually store.

## Local usage

After starting Docker Compose:

```text
http://127.0.0.1:8001/docs
```

Open `GET /monitoring/predictions` and choose a window between 1 and 168 hours.

The endpoint returns `insufficient_data` until both the current and previous windows contain predictions.
