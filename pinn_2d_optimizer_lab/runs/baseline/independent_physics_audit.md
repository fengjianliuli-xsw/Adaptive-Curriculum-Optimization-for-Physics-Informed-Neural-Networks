# Independent physical audit

Post-training checks on 1024 held-out Sobol points; never used to update or choose models.

| PDE | Optimizer | Held-out PDE MSE | Held-out constitutive MSE | Predicted pressure force (N) | Predicted bottom reaction (N) | Predicted force imbalance |
|---|---|---:|---:|---:|---:|---:|
| heat2d | adam | 3.4057e-03 | — | — | — | — |
| heat2d | momentum | 2.4599e-02 | — | — | — | — |
| heat2d | rmsprop | 1.7980e-02 | — | — | — | — |
| heat2d | lbfgs | 5.5928e-05 | — | — | — | — |
| heat2d | adam_lbfgs | 5.1465e-05 | — | — | — | — |
| heat2d | lm | 9.4089e-07 | — | — | — | — |
| heat2d | multiadam | 1.5064e-01 | — | — | — | — |
| heat2d | pcgrad | 9.7858e-02 | — | — | — | — |
| heat2d | config | 4.9900e-01 | — | — | — | — |
| heat2d | soap | 3.0317e-05 | — | — | — | — |
| contact2d | adam | 4.3635e-04 | 2.8219e-03 | 5.07477 | 10.05585 | 98.154% |
| contact2d | momentum | 3.1245e-04 | 3.8127e-03 | 4.58670 | 9.30530 | 102.876% |
| contact2d | rmsprop | 5.8257e-03 | 5.2109e-03 | 5.05499 | 10.22247 | 102.225% |
| contact2d | lbfgs | 1.9057e-04 | 5.9301e-04 | 13.06371 | 12.98477 | 0.604% |
| contact2d | adam_lbfgs | 2.4422e-04 | 9.3521e-04 | 12.85971 | 12.85792 | 0.014% |
| contact2d | lm | 1.0023e-02 | 2.0121e-04 | 13.45050 | 13.36620 | 0.627% |
| contact2d | multiadam | 2.5090e-03 | 1.8618e-02 | 2.20314 | 6.14737 | 179.028% |
| contact2d | pcgrad | 8.0365e-04 | 6.7810e-03 | 3.68939 | 8.28079 | 124.449% |
| contact2d | config | 1.8829e-03 | 5.7349e-04 | 5.80661 | 12.44256 | 114.283% |
| contact2d | soap | 1.3905e-04 | 7.4945e-04 | 13.26749 | 13.13358 | 1.009% |