# Volatility Shape Regime Report: experiment_default

- generated_at: 2026-04-08T17:18:06Z
- asset: SPY
- rows: 3586

## Best feature/embedding/clustering combo
- embedding method: pca
- clustering method: kmeans
- K: 3
- silhouette: 0.20559449699605678

## Predictability summary
- best model: markov_order1
- best accuracy: 0.9925
- persistence accuracy: 0.9925
- fraction S_t+1 = S_t (test): 0.9925
- accuracy on change events only: 0.0000

## Cluster stability / interpretability
- average dwell length: 131.00
- transition entropy: 0.1122

## Key states
- state 0: rising choppy vol, prevalence=0.311, avg_vol=0.01187, future_vol=0.01184
- state 1: low smooth vol, prevalence=0.670, avg_vol=0.00712, future_vol=0.00715
- state 2: stressed shock-decay vol, prevalence=0.019, avg_vol=0.02892, future_vol=0.02837

## Interpretation archetype check
- low smooth vol found: True
- rising/choppy vol found: True
- stressed shock-decay vol found: True

## K=4/K=5 and GMM comparison
- gmm K=3: markov_acc=0.9689, change_acc=0.0000, same_frac=0.9689, silhouette=0.1593
- kmeans K=5: markov_acc=0.9670, change_acc=0.0000, same_frac=0.9670, silhouette=0.1564
- kmeans K=4: markov_acc=0.9670, change_acc=0.0000, same_frac=0.9670, silhouette=0.1509
- gmm K=4: markov_acc=0.9407, change_acc=0.0000, same_frac=0.9407, silhouette=0.1159
- gmm K=5: markov_acc=0.9256, change_acc=0.0000, same_frac=0.9256, silhouette=0.1080
- gmm K=6: markov_acc=0.9049, change_acc=0.0000, same_frac=0.9049, silhouette=0.1049

## Conclusion
Vol-shape states did not beat persistence in this run; transitions remain partly unpredictable.