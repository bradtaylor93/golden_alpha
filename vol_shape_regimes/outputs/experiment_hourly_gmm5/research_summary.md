# Volatility Shape Regime Report: experiment_hourly_gmm5

- generated_at: 2026-04-08T17:57:54Z
- asset: SPY
- rows: 5066

## Best feature/embedding/clustering combo
- embedding method: pca
- clustering method: gmm
- K: 5
- silhouette: 0.10607838473204949

## Predictability summary
- best model: markov_order1
- best accuracy: 0.9455
- persistence accuracy: 0.9455
- fraction S_t+1 = S_t (test): 0.9455
- accuracy on change events only: 0.0000

## Cluster stability / interpretability
- average dwell length: 35.09
- transition entropy: 0.1499

## Key states
- state 0: low smooth vol, prevalence=0.489, avg_vol=0.00246, future_vol=0.00246
- state 1: stressed shock-decay vol, prevalence=0.026, avg_vol=0.00838, future_vol=0.00827
- state 2: rising choppy vol, prevalence=0.151, avg_vol=0.00334, future_vol=0.00335
- state 3: stressed shock-decay vol, prevalence=0.091, avg_vol=0.00480, future_vol=0.00479
- state 4: low smooth vol, prevalence=0.243, avg_vol=0.00301, future_vol=0.00301

## Interpretation archetype check
- low smooth vol found: True
- rising/choppy vol found: True
- stressed shock-decay vol found: True

## K=4/K=5 and GMM comparison
- gmm K=5: markov_acc=0.9455, change_acc=0.0000, same_frac=0.9455, silhouette=0.1061

## Conclusion
Vol-shape states did not beat persistence in this run; transitions remain partly unpredictable.