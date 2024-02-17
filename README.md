# PGPS

This is the repositry for the code of the paper **Path Guided Particle-based Sampling**. 

The contributions of this work are threefold:

1. We propose PGPS as a **novel framework of flow-based sampling methods** and derive a tractable criterion for any differentiable partition-free path in Proposition 3.1 in the paper;
2. We theoretically show that the Wasserstein distance between the target distribution and the PGPS generated distribution following the NN-learned vector field with approximation error $\delta$ and discretization error by step-size $h$ is bounded by $\mathcal{O}(\delta) + \mathcal{O}(\sqrt{h})$ in Theorem 4.2;
3. We experimentally verify the superior performance of the proposed approach over the state-of-the-art benchmarks, in terms of the sampling quality of faster mode seeking and more accurate weight estimating, and the inference quality with **higher testing accuracy** and **stronger calibration ability** in Bayesian inference. The code to reproduce the experiment results reported in the paper is uploaded in the "experiments" folder.



## Illustrative example
Here is an illustrative example showing the effectiveness of PGPS, please refer to the file example.ipynb in the root folder for the code generating this example. 

### Target

![](./independent.png)

### LD over iterations

![](./LD.gif)

### PGPS over iterations

![](./PGPS.gif)

### PGPS evolved time t change with steps

![](./pgps_time.png)