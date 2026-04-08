# Volatility Shape Regime Report: experiment_default

- generated_at: 2026-04-08T16:56:29Z
- asset: SPY
- rows: 3586

## Best feature/embedding/clustering combo
- embedding method: pca
- clustering method: kmeans
- K: 3
- silhouette: 0.20533975023377107

## Predictability summary
- best model: conditional_multinomial_logit
- best accuracy: 0.9934
- persistence accuracy: 0.9925
- fraction S_t+1 = S_t (test): 0.9925
- accuracy on change events only: 0.0000

## Cluster stability / interpretability
- average dwell length: 131.00
- transition entropy: 0.1122

## Key states
- state 0: rising choppy vol, prevalence=0.311, avg_vol=0.01187, future_vol=0.01183
- state 1: low smooth vol, prevalence=0.670, avg_vol=0.00712, future_vol=0.00715
- state 2: stressed shock-decay vol, prevalence=0.019, avg_vol=0.02892, future_vol=0.02837

## Interpretation archetype check
- low smooth vol found: True
- rising/choppy vol found: True
- stressed shock-decay vol found: True

## K=4/K=5 and GMM comparison
- kmeans K=5: markov_acc=0.9670, change_acc=0.0000, same_frac=0.9670, silhouette=0.1563
- kmeans K=4: markov_acc=0.9670, change_acc=0.0000, same_frac=0.9670, silhouette=0.1511
- gmm K=3: markov_acc=0.9539, change_acc=0.0000, same_frac=0.9539, silhouette=0.1604
- gmm K=5: markov_acc=0.9379, change_acc=0.0000, same_frac=0.9379, silhouette=0.1090
- gmm K=4: markov_acc=0.9360, change_acc=0.0000, same_frac=0.9360, silhouette=0.1148
- gmm K=6: markov_acc=0.9134, change_acc=0.0000, same_frac=0.9134, silhouette=0.0951

## Conclusion
Vol-shape states improved next-state prediction versus persistence baseline.