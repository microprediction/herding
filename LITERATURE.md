
## Entropic Herding  [arxiv](https://arxiv.org/abs/2112.11616)
Hiroshi Yamashita, Hideyuki Suzuki, Kazuyuki Aihara

Herding is a deterministic algorithm used to generate data points that can be regarded as random samples satisfying input moment conditions. The algorithm is based on the complex behavior of a high-dimensional dynamical system and is inspired by the maximum entropy principle of statistical inference. In this paper, we propose an extension of the herding algorithm, called entropic herding, which generates a sequence of distributions instead of points. Entropic herding is derived as the optimization of the target function obtained from the maximum entropy principle. Using the proposed entropic herding algorithm as a framework, we discuss a closer connection between herding and the maximum entropy principle. Specifically, we interpret the original herding algorithm as a tractable version of entropic herding, the ideal output distribution of which is mathematically represented. We further discuss how the complex behavior of the herding algorithm contributes to optimization. We argue that the proposed entropic herding algorithm extends the application of herding to probabilistic modeling. In contrast to original herding, entropic herding can generate a smooth distribution such that both efficient probability density calculation and sample generation become possible. To demonstrate the viability of these arguments in this study, numerical experiments were conducted, including a comparison with other conventional methods, on both synthetic and real data.

## Super-Samples from Kernel Herding [arxiv](https://arxiv.org/abs/1203.3472)
Yutian Chen, Max Welling, Alex Smola

We extend the herding algorithm to continuous
spaces by using the kernel trick. The resulting
“kernel herding” algorithm is an infinite memory deterministic process that learns to approximate a PDF with a collection of samples. We
show that kernel herding decreases the error of
expectations of functions in the Hilbert space at
a rate O(1/T ) which is much faster than the usual
O(1/T) for iid random samples. We illustrate
kernel herding by approximating Bayesian predictive distributions.

## Optimally-Weighted Herding is Bayesian Quadrature [pdf](https://arxiv.org/abs/1204.1664)
Ferenc Huszár, David Duvenaud

Herding and kernel herding are deterministic methods of choosing samples which summarise a probability distribution. A related task is choosing samples for estimating integrals using Bayesian quadrature. We show that the criterion minimised when selecting samples in kernel herding is equivalent to the posterior variance in Bayesian quadrature. We then show that sequential Bayesian quadrature can be viewed as a weighted version of kernel herding which achieves performance superior to any other weighted herding method. We demonstrate empirically a rate of convergence faster than O(1/N). Our results also imply an upper bound on the empirical error of the Bayesian quadrature estimate.

## Sparse solutions of the kernel herding algorithm by improved gradient approximation [arxiv](https://arxiv.org/abs/2105.07900)
Kazuma Tsuji, Ken'ichiro Tanaka

The kernel herding algorithm is used to construct quadrature rules in a reproducing kernel Hilbert space (RKHS). While the computational efficiency of the algorithm and stability of the output quadrature formulas are advantages of this method, the convergence speed of the integration error for a given number of nodes is slow compared to that of other quadrature methods. In this paper, we propose a modified kernel herding algorithm whose framework was introduced in a previous study and aim to obtain sparser solutions while preserving the advantages of standard kernel herding. In the proposed algorithm, the negative gradient is approximated by several vertex directions, and the current solution is updated by moving in the approximate descent direction in each iteration. We show that the convergence speed of the integration error is directly determined by the cosine of the angle between the negative gradient and approximate gradient. Based on this, we propose new gradient approximation algorithms and analyze them theoretically, including through convergence analysis. In numerical experiments, we confirm the effectiveness of the proposed algorithms in terms of sparsity of nodes and computational efficiency. Moreover, we provide a new theoretical analysis of the kernel quadrature rules with fully-corrective weights, which realizes faster convergence speeds than those of previous studies

## Sampling Over Riemannian Manifolds Using Kernel Herding [pdf](https://openreview.net/pdf?id=f30VKPZMBP)
Sandesh Adhikary1 and Byron Boots1

Kernel herding is a deterministic sampling algorithm
designed to draw ‘super samples’ from probability distributions
when provided with their kernel mean embeddings in a reproducing
kernel Hilbert space (RKHS). Empirical expectations of functions
in the RKHS formed using these super samples tend to converge
even faster than random sampling from the true distribution itself.
Standard implementations of kernel herding have been restricted
to sampling over flat Euclidean spaces, which is not ideal for
applications such as robotics where more general Riemannian manifolds may be appropriate. We propose to adapt kernel herding to
Riemannian manifolds by (1) using geometry-aware kernels that
incorporate the appropriate distance metric for the manifold and (2)
using Riemannian optimization to constrain herded samples to lie
on the manifold. We evaluate our approach on problems involving
various manifolds commonly used in robotics including the SO(3)
manifold of rotation matrices, the spherical manifold used to encode
unit quaternions, and the manifold of symmetric positive definite
matrices. We demonstrate that our approach outperforms existing
alternatives on the task of resampling from empirical distributions
of weighted particles, a problem encountered in applications such as
particle filtering. We also demonstrate how Riemannian kernel herding can be used as part of the kernel recursive approximate Bayesian
computation algorithm to estimate parameters of black-box simulators, including inertia matrices of an Adroit robot hand simulator.
Our results confirm that exploiting geometric information through
our approach to kernel herding yields better results than alternatives
including standard kernel herding with heuristic projections.

## Generalized Coverage for More Robust Low-Budget Active Learning  [pdf](https://arxiv.org/pdf/2407.12212)
Wonho Bae, Junhyug Noh and Danica J. Sutherland

The ProbCover method of Yehuda et al. is a well-motivated
algorithm for active learning in low-budget regimes, which attempts to
“cover” the data distribution with balls of a given radius at selected data
points. We demonstrate, however, that the performance of this algorithm
is extremely sensitive to the choice of this radius hyper-parameter, and
that tuning it is quite difficult, with the original heuristic frequently failing. We thus introduce (and theoretically motivate) a generalized notion
of “coverage,” including ProbCover’s objective as a special case, but also
allowing smoother notions that are far more robust to hyper-parameter
choice. We propose an efficient greedy method to optimize this coverage,
generalizing ProbCover’s algorithm; due to its close connection to kernel
herding, we call it “MaxHerding.” The objective can also be optimized
non-greedily through a variant of
k-medoids, clarifying the relationship
to other low-budget active learning methods. In comprehensive experiments, MaxHerding surpasses existing active learning methods across
multiple low-budget image classification benchmarks, and does so with
less computational cost than most competitive methods.


## On the Global Linear Convergence of Frank-Wolfe Optimization Variants [pdf](https://arxiv.org/pdf/1511.05932)
Simon Lacoste-Julien, Martin Jaggi

The Frank-Wolfe (FW) optimization algorithm has lately re-gained popularity
thanks in particular to its ability to nicely handle the structured constraints appearing in machine learning applications. However, its convergence rate is known
to be slow (sublinear) when the solution lies at the boundary. A simple lessknown fix is to add the possibility to take ‘away steps’ during optimization, an
operation that importantly does not require a feasibility oracle. In this paper, we
highlight and clarify several variants of the Frank-Wolfe optimization algorithm
that have been successfully applied in practice: away-steps FW, pairwise FW,
fully-corrective FW and Wolfe’s minimum norm point algorithm, and prove for
the first time that they all enjoy global linear convergence, under a weaker condition than strong convexity of the objective. The constant in the convergence rate
has an elegant interpretation as the product of the (classical) condition number of
the function with a novel geometric quantity that plays the role of a ‘condition
number’ of the constraint set. We provide pointers to where these algorithms have
made a difference in practice, in particular with the flow polytope, the marginal
polytope and the base polytope for submodular optimization.
