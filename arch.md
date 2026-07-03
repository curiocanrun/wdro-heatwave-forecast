Wasserstein Distributionally Robust Optimization - Mathematical Formulation

## The Problem

We want a model that performs well on future climate data, but we only have historical observations.

Let:
- $Z$ = weather features (temperature, pressure, humidity, wind)
- $Y$ = heatwave label (1 if heatwave, 0 otherwise)
- $l(\theta, Z)$ = loss function (binary cross-entropy)

Standard Empirical Risk Minimization (ERM):

$$ \min_{\theta} \mathbb{E}_{Z \sim \hat{P}_n}[l(\theta, Z)] $$

where $\hat{P}_n$ is the empirical distribution of historical data.

## The Issue

$\hat{P}_n$ has zero density except at observed points. If future climate shifts, the model cannot perform well.

## W-DRO Formulation

Instead of trusting $\hat{P}_n$ alone, we define a ball of distributions around it and minimize the worst-case expected loss:

$$ \min_{\theta} \sup_{\mathbb{Q} : W_1(\hat{P}_n, \mathbb{Q}) \le \epsilon} \mathbb{E}_{Z \sim \mathbb{Q}}[l(\theta, Z)] $$
where $W_1$ is the 1-Wasserstein distance and $\epsilon$ is the Wasserstein radius.

**In our context:** Instead of trusting only the past, we consider all possible future distributions close to the past and train against the worst one.

## Wasserstein Distance


For two distributions $P$ and $\mathbb{Q}$:

$$ W_1(P, \mathbb{Q}) = \inf_{\pi \in \Pi(P, \mathbb{Q})} \int_{\mathcal{Z} \times \mathcal{Z}} \|z - \tilde{z}\| \, d\pi(z, \tilde{z}) $$

This is the minimum cost of transporting probability mass from $P$ to $\mathbb{Q}$.

## Strong Duality


The infinite-dimensional inner maximization has a finite-dimensional dual:

$$ \sup_{\mathbb{Q} : W_1(\hat{P}_n, \mathbb{Q}) \le \epsilon} \mathbb{E}_{\mathbb{Q}}[l(\theta, Z)] = $$

$$ \inf_{\lambda \ge 0} \left\{ \lambda \epsilon + \frac{1}{n} \sum_{i=1}^n \sup_{Z \in \mathcal{Z}} \left[ l(\theta, Z) - \lambda \|Z - Z_i\| \right] \right\} $$

where $\lambda$ is the Lagrange multiplier for the Wasserstein constraint.

## The Saddle-Point Problem

$$ \min_{\theta} \max_{\lambda \ge 0} \left\{ \lambda \epsilon + \frac{1}{n} \sum_{i=1}^n \max_{Z} \left[ l(\theta, Z) - \lambda \|Z - Z_i\| \right] \right\} $$

Three components:
- Outer min ($\theta$): model weights
- Inner max ($\lambda$): dual variable (penalty per unit of transport)
- Innermost max ($Z$): adversarial input

## Numerical Approximation


For neural networks, the innermost $\max_Z$ is solved iteratively using Projected Gradient Descent (PGD):

$$ Z^{(t+1)} = Z^{(t)} + \alpha \cdot \text{sign}\left( \nabla_Z \left[ l(\theta, Z^{(t)}) - \lambda \|Z^{(t)} - Z_i\| \right] \right) $$

## The Guarantee

If the true future distribution $\mathbb{Q}_{future}$ satisfies $W_1(\hat{P}_n, \mathbb{Q}_{future}) \le \epsilon$, then:

$$ \mathbb{E}_{\mathbb{Q}_{future}}[l(\theta^*_{DRO}, Z)] \le \mathbb{E}_{\hat{P}_n}[l(\theta^*_{DRO}, Z)] + \epsilon \cdot \text{Lip}(l) $$

where $\text{Lip}(l)$ is the Lipschitz constant of the loss.

Future performance is bounded by historical performance plus the physical shift multiplied by model sensitivity.
