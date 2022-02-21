# ves-openmm-torch

## Flowchart of options

```mermaid
graph LR;
    1[Choose Basis]-->A(Legendre polynomial)-->AA[Choose basis size];
    1-->B(Neural network)-->BB[Choose network architecture];

    2[Choose Target]-->C(Uniform distribution);
    2-->D(Well-tempered distribution)-->D1[Choose bias factor];
    D-->D2[Choose free energy localization]
    D2-->D2A(No localization)
    D2-->D2B(Localized with sigmoidal cutoff)

    3(Update)-->E[Choose averaging method]
    E-->E1(Instantaneous)
    E-->E2(Running average)
    E-->E3(Weighted running average)-->E3A[Choose exponent]
    E-->E4(Moving window average)-->E4A[Choose window size]

    3-->F[Choose optimizer]
    F-->F1(Stochastic PyTorch optimizers)
    F-->F2(Original VES 2nd order optimizer)

    3-->G[Choose learning rate]
    G-->G1[Choose learning rate decay]
    G1-->G1A(No decay)
    G1-->G1B(Decay below KL-divergence cutoff)
    G1B-->G1B1[Choose cutoff]
    G1B-->G1B2[Choose exponent]
 ```
