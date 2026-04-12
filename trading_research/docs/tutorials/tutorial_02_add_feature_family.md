# Tutorial 02 - Add a Feature Family

1. Implement a class with `name`, `version`, and `build(...)`:

```python
class MyFeatureFamily:
    name = "my_family"
    version = "1"

    def build(self, bars, params):
        out = bars[["timestamp", "asset"]].copy()
        out["my_signal"] = bars.groupby("asset")["close"].pct_change().fillna(0.0)
        return out
```

2. Register it:

```python
from trading_research.features import FeatureRegistry

registry = FeatureRegistry()
registry.register(MyFeatureFamily())
```

3. Reference in a `FeatureNode`:

```python
FeatureNode(name="features_custom", family_name="my_family", bars_inputs=["bars"])
```

Because features are explicit artifacts, your new family is audit-visible and reusable.
