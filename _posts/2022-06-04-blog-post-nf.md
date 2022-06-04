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

Let $\bm{z}\in\mathbb{R}^d$ has density $p_1(\bm{z})$ and $\bm{x}=f(\bm{z})$ is an invertible and differentiable transformation ($\bm{x}$ needs to have the same dimension as $\bm{z}$). Then $\bm{x}$ has density
$$p_2(\bm{x})=p_1(f^{-1}(\bm{x}))\cdot\left|\det\left(\frac{\partial f^{-1}(\bm{x})}{\partial \bm{x}}\right)\right|=p_1(f^{-1}(\bm{x}))\cdot\left|\det\left(\frac{\partial f(\bm{z})}{\partial \bm{z}}\right)^{-1}\right|$$.

(Intuitively, $\int p_2(x)dx=\int p_1(f^{-1}(x))dx/dz\cdot dz$)
The input and output space of the mapping should have the same dimension. If $d=1$, , it is sufficient that $f$ is strictly monotonic. Ideally both $f$ and $f^{-1}$ are are continuously differentiable. Differentiability is a sufficient condition, and in theory the mapping $f$ does not have to be differentiable everywhere (i.e., piecewise continuous). In practice we usually use only differentiable transformation.

We can also define a series of stacking transformations $f_i$ such that $\bm{x}=\bm{z}_K=f_K(f_{K-1}(\cdots f_2(f_1(\bm{z_0}))))$. The change of variables formula gives
$$p_K(\bm{x})=p_0(\bm{z})\prod_{i=1}^K\left|\det\left(\frac{\partial f_i^{-1}(\bm{z}_i)}{\partial \bm{z}_i}\right)\right|$$,

or the log version
$$\log p_K(\bm{x})=\log p_0(\bm{z})+\sum_{i=1}^K\log\left|\det\left(\frac{\partial f_i^{-1}(\bm{z}_i)}{\partial \bm{z}_i}\right)\right|$$.

2.Forward and Reverse Parametrization
======

2.1 Reverse Parametrization
------
Reverse Parametrization is for density evaluation at $\bm{x}$. If we know $g=f^{-1}$ analytically, $g_\psi(\bm{x})=\bm{z}$ is computable and parameter $\psi$ can be learned. The density can be evaluated by
$$p_2(\bm{x})=p_1(g_\psi(\bm{x}))\cdot\left|\det\left(\frac{\partial g_\psi(\bm{x})}{\partial \bm{x}}\right)\right|$$

This formula also works for the satcking transformation $g_\psi=g_{\psi_1}\circ\ \cdots\circ g_{\psi_k}$.

To learn the parameter $\psi$:
$$\max\limits_{\psi}p_\psi(\mathcal{D})=\max\limits_{\psi}\frac{1}{n}\sum_{\bm{x}^{(j)}\in\mathcal{D}}\log p_\psi(\bm{x}^{(j)})$$.

2.2 Forward Parametrization
------

Forward Parametrization can be used for sampling $\bm{x}$ and density estimation. We assume that $f_\theta(\bm{z})=\bm{x}$ is computable and parameter $\theta$ can be learned. The inverse $f^{-1}$ exists but may not be easy to compute.

For each sample $\bm{z}^{(j)}\sim p_1(\bm{z})$, we can compute the sample $$\bm{x}^{(j)}=f_\theta(\bm{z}^{(j)})\sim p_2(\bm{x})$$ and $$p_2(\bm{x}^{(j)})=p_1(\bm{z}^{(j)})\cdot\left|\det\left(\frac{\partial f_\theta(\bm{z}^{(j)})}{\partial \bm{z}}\right)\right|^{-1}$$.

The formula also works for $f_\theta=f_{\theta_1}\circ\ \cdots\circ f_{\theta_k}$.

PS: This is exactly what we need in Variational Inference:
to sample $\bm{x}$ from a distribution $q$ and to compute the probability $q(\bm{x})$ for this sample.


References
------

Machine Learning for Graphs and Sequential Data (IN2323), TUM S21 


<!-- Aren't headings cool?
<!------>
