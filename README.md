# ves-openmm-torch

## Flowchart of options

```mermaid
graph LR;
    1[Choose Basis]-->A(Legendre polynomial)-->AA[Choose basis size];
    1-->B(Neural network)-->BB[Choose network architecture];

    2[Choose Target]-->C(Uniform distribution);
    2-->D(Well-tempered distribution)-->D1[Choose bias factor];
    D-->D2[Choose localization]
    D2-->D2A(No localization)
    D2-->D2B(Localized with sigmoidal cutoff)

    3(Update)-->E[Choose averaging method]
    E-->E1(Instantaneous)
    E-->E2(Running average)
    E-->E3(Weighted running average)-->E3A[Choose exponent]
    E-->E4(Moving window average)-->E4A[Choose window size]
    3-->F[Choose optimizer]
    F-->F1(Torch optimizers)
    F-->F2(VES 2nd order optimizer)
 ```
