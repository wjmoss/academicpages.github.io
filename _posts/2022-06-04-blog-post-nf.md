---
title: 'Basic Normalizing Flow'
date: 2022-06-04
permalink: /posts/2022/06/nf/
tags: 
  - machine learning
  - 

---

1.Change of variables formula
======
Normalizing Flows (NF) can model flexible distributions for data
sampling and density estimation. The idea is based on change of variables formula, applying a series of transformations on a simple distribution to approximate a complex distribution.

Let $\boldsymbol{z}\in\mathbb{R}^d$ has density $p_1(\boldsymbol{z})$ and $\boldsymbol{x}=f(\boldsymbol{z})$ is an invertible and differentiable transformation ($\boldsymbol{x}$ needs to have the same dimension as $\boldsymbol{z}$). Then $\boldsymbol{x}$ has density
$$p_2(\boldsymbol{x})=p_1(f^{-1}(\boldsymbol{x}))\cdot\left|\det\left(\frac{\partial f^{-1}(\boldsymbol{x})}{\partial \boldsymbol{x}}\right)\right|=p_1(f^{-1}(\boldsymbol{x}))\cdot\left|\det\left(\frac{\partial f(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{-1}\right|$$.

(Intuitively, $\int p_2(x)dx=\int p_1(f^{-1}(x))dx/dz\cdot dz$)
The input and output space of the mapping should have the same dimension. If $d=1$, , it is sufficient that $f$ is strictly monotonic. Ideally both $f$ and $f^{-1}$ are are continuously differentiable. Differentiability is a sufficient condition, and in theory the mapping $f$ does not have to be differentiable everywhere (i.e., piecewise continuous). In practice we usually use only differentiable transformation.

We can also define a series of stacking transformations $f_i$ such that $\boldsymbol{x}=\boldsymbol{z}_K=f_K(f_{K-1}(\cdots f_2(f_1(\boldsymbol{z_0}))))$. The change of variables formula gives
$$p_K(\boldsymbol{x})=p_0(\boldsymbol{z})\prod_{i=1}^K\left|\det\left(\frac{\partial f_i^{-1}(\boldsymbol{z}_i)}{\partial \boldsymbol{z}_i}\right)\right|$$,

or the log version
$$\log p_K(\boldsymbol{x})=\log p_0(\boldsymbol{z})+\sum_{i=1}^K\log\left|\det\left(\frac{\partial f_i^{-1}(\boldsymbol{z}_i)}{\partial \boldsymbol{z}_i}\right)\right|$$.

2.Forward and Reverse Parametrization
======

2.1 Reverse Parametrization
------
Reverse Parametrization is for density evaluation at $\boldsymbol{x}$. If we know $g=f^{-1}$ analytically, $g_\psi(\boldsymbol{x})=\boldsymbol{z}$ is computable and parameter $\psi$ can be learned. The density can be evaluated by
$$p_2(\boldsymbol{x})=p_1(g_\psi(\boldsymbol{x}))\cdot\left|\det\left(\frac{\partial g_\psi(\boldsymbol{x})}{\partial \boldsymbol{x}}\right)\right|$$

This formula also works for the satcking transformation $g_\psi=g_{\psi_1}\circ\ \cdots\circ g_{\psi_k}$.

To learn the parameter $\psi$:
$$\max\limits_{\psi}p_\psi(\mathcal{D})=\max\limits_{\psi}\frac{1}{n}\sum_{\boldsymbol{x}^{(j)}\in\mathcal{D}}\log p_\psi(\boldsymbol{x}^{(j)})$$.

2.2 Forward Parametrization
------

Forward Parametrization can be used for sampling $\boldsymbol{x}$ and density estimation. We assume that $f_\theta(\boldsymbol{z})=\boldsymbol{x}$ is computable and parameter $\theta$ can be learned. The inverse $f^{-1}$ exists but may not be easy to compute.

For each sample $\boldsymbol{z}^{(j)}\sim p_1(\boldsymbol{z})$, we can compute the sample $$\boldsymbol{x}^{(j)}=f_\theta(\boldsymbol{z}^{(j)})\sim p_2(\boldsymbol{x})$$ and $$p_2(\boldsymbol{x}^{(j)})=p_1(\boldsymbol{z}^{(j)})\cdot\left|\det\left(\frac{\partial f_\theta(\boldsymbol{z}^{(j)})}{\partial \boldsymbol{z}}\right)\right|^{-1}$$.

The formula also works for $f_\theta=f_{\theta_1}\circ\ \cdots\circ f_{\theta_k}$.

PS: This is exactly what we need in Variational Inference:
to sample $\boldsymbol{x}$ from a distribution $q$ and to compute the probability $q(\boldsymbol{x})$ for this sample.



References
------

Machine Learning for Graphs and Sequential Data (IN2323), TUM S21 


<!-- Aren't headings cool?
<!------>
