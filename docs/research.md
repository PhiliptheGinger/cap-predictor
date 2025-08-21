# Research Hooks

The research package exposes a light‑weight interface for experimenting with new strategies.

```python
from sentimental_cap_predictor.research.idea_schema import Idea
from sentimental_cap_predictor.research.apply_idea import apply_idea

def backtest(idea: Idea):
    # user provided logic
    return idea.params

idea = Idea(name="My idea", params={"window": 5})
result = apply_idea(idea, backtest)
```

## Experiment Tracking

The `LocalTracker` offers lightweight experiment logging without requiring an
external service. Parameters can be recorded alongside metrics and later joined
for analysis.

```python
from research.tracking import LocalTracker

tracker = LocalTracker(root="./runs")
tracker.start_run({"window": 5})
tracker.log_metrics({"accuracy": 0.9})
tracker.end_run()
```

## Additional Resources

- [User Manual](user_manual.md) – end-to-end usage guide
- [Documentation Index](index.md) – list of all documentation topics
