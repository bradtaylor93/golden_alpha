# Volatility Shape Regime Report: experiment_default

- generated_at: 2026-04-08T16:27:46Z
- asset: SPY
- rows: 3586

## Best feature/embedding/clustering combo
- embedding method: pca
- clustering method: kmeans
- K: 3
- silhouette: 0.2057280731498028

## Predictability summary
- best model: markov_order1
- best accuracy: 0.9925
- persistence accuracy: 0.9925

## Cluster stability / interpretability
- average dwell length: 131.00
- transition entropy: 0.1122

## Key states
- state 0: high clustered vol, falling, decay, prevalence=0.312, avg_vol=0.01187, future_vol=0.01183
- state 1: medium clustered vol, flat, escalation, prevalence=0.669, avg_vol=0.00711, future_vol=0.00714
- state 2: high clustered vol, rising, decay, prevalence=0.019, avg_vol=0.02892, future_vol=0.02837

## Conclusion
Vol-shape states did not beat persistence in this run; transitions remain partly unpredictable.