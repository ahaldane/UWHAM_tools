
Simple script implementing UWHAM (Unbinned Weighted Histogram Analysis Method) in Python

## Usage

### Command-line Usage

The basic functionality is accessible by running the script from the command line.

```
usage: UWHAM_tools.py [-h] [--weights WEIGHTS] [--tol TOL] [--niter NITER]
                      [BE ...]

Perform UWHAM Analysis

positional arguments:
  BE                 Boltzmann factor files

options:
  -h, --help         show this help message and exit
  --weights WEIGHTS  save UWHAM weights to this file
  --tol TOL          change in Fs at which to stop iteration
  --niter NITER      If given, ignore tol and iterate this many times
```

"BE" should be a set of numpy ".npy" array files of negative log-likelihoods, in
order:

    BE_S1H1 BE_S2H1 BE_S3H1 ...  BE_S1H2 BE_S2H2 BE_S3H2 ...

where BE_SiHj is the array of negative log-likelihoods for the sample-set
generated by potential $i$ evaluated using potential $j$. In a common
scenario in physics, the negative log-likelihood is equal to the Boltzmann
factor $B_j E_j(x)$ for inverse temperature $B_j$ and Hamiltonian "energy"
$E_j(x)$ for sample $x$.

The number of "BE" input arrays should be a perfect square $N^2$ as there
should be one set of samples from each of $N$ potentials, each evaluated
for each of the $N$ potentials.

The script will print out:

    "Ns = ..."
        The number of raw input samples from each Hamiltonian, for reference 
    "Neff = ..."
        The effective number of samples corresponding to the computed UWHAM
        weights for each Hamiltonian 
    "ΔlogZ = ..."
        The relative log partition functions of each Hamiltonian, relative
        to the mean. (i.e, relative free energies of each Hamiltonian)

If an output file is supplied with the --weights option, the UWHAM weights
for all samples in all Hamiltonians will be saved to file. This will be an
array of shape (N,M) for N Hamiltonians, and M total samples which are the
input samples concatenated together. Use this for weighted averages:

    >>> weights = np.load('weights.npy')
    >>> Vbar1 = np.average(V, weights=w[0]) # average under 0th potential

### Programmatic Usage

The script is also importable in python and provides the following additional methods:

* `UWHAM(BE, tol=1e-6, niter=None)`  
  UWHAM for an arbitrary set of Hamiltonians and samples, given the `B*E` log-likelihood values evaluated for each sample for each Hamiltonian.

* `UWHAM_generalized(samples, Hamiltonians, tol=1e-6, niter=None)`  
  UWHAM for an arbitrary set of Hamiltonians, given the callable Hamiltonian
  functions and a set of samples from each Hamiltonian.

* `Neff(weights)`  
  An estimate of the effective number of samples corresponding to this set of
  weights.

* `UWHAM_manyPT(B, E, tol=1e-6, niter=None, Bt=None)`  
  UWHAM for the same Hamiltonian at a large number of different temperatures,
  given the E of each sample and B it was generated with.

 * `manyPT_weights(Bt, Si, E)`
   UWHAM weights for a target temperature Bt for a manyPT dataset

## Theoretical Background

### Derivation of the UWHAM equations

Given samples from multiple likelihood functions (i.e., Hamiltonians and temperatures), UWHAM allows us to estimate the
partition function ratio between the different Hamiltonians, among other quantities, as well as to evaluate ensemble averages by combining data from multiple simulations, giving increased statistical power.

Consider a set of M different likelihood functions $a \in 1..M$, from which we obtain samples $u$.
Often, the log of the likelihood is the product of an inverse temperature $B_a$ and a Hamiltonian (energy) $E_a(u)$. Let us assume this here without loss of generality.  Define $q_a(u) = B_a E_a(u)$ is the Boltzmann factor (negative log likelihood) for state $u$ in potential $a$.  The partition function for potential $a$ is $Z_a = \sum_u e^{-q_a(u)}$ summed over all possible states $u$.

From each $a$, we are given set of $N_a$ samples drawn from probability distribution $p_a(u) = e^{-q_a(u)}/Z_a$. In total, we have $N = \sum_a N_a$ samples.

UWHAM arises from trying to use this data to estimate the probability of
the observed states in another (potentially unsampled) Hamiltonian, the
"target" $t$. It can be derived through a maximum-likelihood analysis given the
data.  The likelihood of the dataset, in the form of counts $n_{au}$ for observations of state $u$ (typically 1) in Hamiltonian $a$, is:

```math
  \mathcal{L}(n|p) = \prod_a \text{Binom}(\{n_{a\cdot}\}) \prod_u p_a(u)^{n_{au}}
```

We can use a "reweighting" identity to relate $p_t(u)$ to $p_a(u)$:

```math
\begin{aligned}
p_a(u) &= \frac{e^{-q_a(u)}}{Z_a} \frac{p_t(u)}{e^{-q_t(u)}/Z_t} \\
       &= \frac{e^{q_t(u) - q_a(u)}}{Z_t/Z_a} p_t(u)  \equiv \frac{r_{at}(u)}{ R_{at}} p_t(u)
\end{aligned}
```

where we can easily evaluate the difference in Boltzmann factors $r_{at}(u) = e^{q_t(u) - q_a(u)}$ (i.e., ratio of likelihoods), but the ratio of partition functions $R_{at} = Z_t/Z_a$ is not
available to us as it naively requires a sum over all states. Note that by normalization we must have:

```math
  R_{at} = \sum_u r_{at}(u) p_t(u)
```

We can use these relations to write the dataset-likelihood $\mathcal{L}$ as a function of $p_t(u)$.
We then maximize $\log \mathcal{L}$ as a function of $p_t(u)$ with the normalization constraint that $\sum_u p_t(u) = 1$. We find that although we cannot isolate
$p_t(u)$ algebraically, the maximum-likelihood solution $^\circ$ satisfies the equation

```math
  p^\circ_t(u) = \frac{\sum_a n_{au}}{\sum_a N_a \frac{r_{at}(u)}{R^\circ_{at}}}
```

with $R^\circ_{at} = \sum_u r_{at}(u) p^\circ_t(u)$ which is a function of $p^\circ_t(u)$, which means this solution is ambiguous by a constant scaling factor (cancels on both sides). This form suggests we can solve for $p^\circ_t(u)$ numerically by solving this equation iteratively.

Note this solution can be interpreted as giving a per-sample weight of

```math
  w_t(i) = \frac{1}{ \sum_a N_a \frac{r_{at}(u_i)}{R^\circ_{at}} } 
```
for each sample $u_i$ in our dataset, for $i \in 1..N$. The Hamiltonian from which sample $i$ was drawn does not factor into this interpretation. Then the estimate of $R^\circ_{at}$ can be seen as a weighted mean over the data:
```math
 R^\circ_{at} = \sum_i r_{at}(u_i) w_t(i)
```

Finally, we introduce a convenient change of variables

```math
   S_i = \log[w_t(i)] + q_t(u_i), \quad \quad   F_a = -\log R^\circ_{at}
```
since then the equations for $p^\circ_t(u)$ and $R^\circ_{at}$ can be rewritten as:

```math
\begin{aligned}
    F_a &= -\log \sum_i e^{S_i - q_a(u_i)} \\
   S_i &= -\log \sum_a e^{F_a - q_a(u_i) + \log N_a}
\end{aligned}
```

in which the target temperature $t$ does not appear, and therefore the solution for $F_a$ and $S_i$ is independent of $t$.
This pair of equations can be solved iteratively. The form of $\log \sum e^x$ in both of these equations allows use of the `logsumexp` function to avoid floating-point overflow and improve numerical precision, and this is used in this module's implementation.

### Interpretation

$F$ gives the log ratio of partition functions, i.e. it can be seen as a
relative free energy. $S$ behaves a bit like an entropy, though it is really related to a
log weight.  Note that if we interpret $R^\circ_{at}$ as a ratio $Z^\circ_t/Z^\circ_a$, the the weights have an interpretation rewritten as:
```math
w_t(i) = \frac{p^\circ_t(u_i)}{\sum_a N_a p^\circ_a(u_i)}
```
which is the probability of state $u_i$ in Hamiltonian $t$ divided by the expected number of times state $u$ will appear in our total dataset. In other words, the weight values serve to avoid double-counting if the state $u$ is likely to appear in the samples from multiple Hamiltonians: If $u$ is only likely under a single Hamiltonian $a$ as $p_a(u) \gg p_b(u)$ for all $b$, then we get $w_a(i) = p^\circ_a(u_i)/(N_a p^\circ_a(u_i)) = 1/N_a$, equivalent to a single count. If $u_i$ can appear in two Hamiltonians $a$ and $b$ with equal probability and no others, the weight will be $1/(N_a + N_b)$, and if the sample counts are equal $N_b = N_a$ the weight is $1/2N_a$, or half a count compared to the previous case, which makes sense because we have a doubled chance of sampling state $u$ in our combined dataset.

By reversing the change of variables above, we can get the weights
$w_t(i)$ for any desired target Hamiltonian $t$, and therefore estimate any
averaged observables under that Hamiltonian by weighted sums over the data
using these weights. 

Since such a weighted average uses data from multiple simulations, it
can have greater statistical power than $N$ samples from a single simulation
with Hamiltonian $t$.  One way to see this is to consider the weighted average of a set of  Bernoulli trials with success probability $p$, i.e. a weighted average of an indicator function for any property of interest of our system where the weights account for double-counting. That is, for weights $w_i$ and Bernouilli random variables $X_i$, we want to compute the variance of $X_\text{tot} = \frac{\sum_i w_i X_i}{\sum_i w_i}$, as a reflection of the statistical uncertainty of this average. Using the scaling and addition properties of variances, we find this variance is 
```math
\text{Var}(X_\text{tot}) = \frac{p(1-p)}{N_\text{eff}} \quad\quad\quad\text{with } \quad N_\text{eff} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
```
This can be compared to the variance of an unweighted average of $N$ Bernoulli trials which is $\frac{p(1-p)}{N}$. The result has the same functional form but with $N$ replaced by $N_\text{eff}$. In other words, the statistical
uncertainty of the weighted average behaves as if averaging
$N_\text{eff}$ unweighted samples. As expected, in the case $w_i =
1$ for all $i$, i.e. the equal-weight case, we get $N_\text{eff} = N$, and in the case one weight is much larger than the others we get $N_\text{eff} = 1$. 

In practice, the $N_\text{eff}$ for estimates using UWHAM will be greater than the $N_t$ samples from the target Hamiltonian, assuming the target was one of the sampled Hamiltonians. The amount of increase of $N_\text{eff}$ over $N_t$ will depend on the "overlap" of the different likelihood functions in state space. The minimum possible value of $N_\text{eff}$ will be $N_t$ if there is no overlap, and its maximum value will be the total number of samples $N = \sum_a N_a$ if all Hamiltonians overlap perfectly.

### References

For a similar derivation using the binomial distribution see:  
Multidimensional Adaptive Umbrella Sampling: Applications to Main Chain and
Side Chain Peptide Conformations. JCC 1998. https://doi.org/10.1002/%28SICI%291096-987X%28199709%2918%3A12%3C1450%3A%3AAID-JCC3%3E3.0.CO%3B2-I 

An alternate derivation is given in:  
Theory of binless multi-state free energy estimation with applications to
protein-ligand binding. J Chem Phys 2012. https://doi.org/10.1063%2F1.3701175

## Example

An example computation is provided in the script in method "test". 

In this demonstration, we consider a Hamiltonian that assigns an energy $E(u)$ to all states $u$ of a system, simulated at two different temperatures with inverse temperatures $1/kT_1 = 1.0$ and $1/kT_2 = \sqrt{2}$. In Bayesian language, this corresponds to two different likelihood functions $\log \mathcal{L}_1(E) = -E/kT_1$, $\log \mathcal{L}_2(E) = -E/kT_2$ for obtaining a sample with value $E$. The density of states (aka the prior, in Bayesian language) is a Gaussian centered at E=0 with standard deviation of 2.

Analytically, one can show this system has the following properties: 
 * $\mathbb{E}_1[E] = -4$, &nbsp;&nbsp;&nbsp; $\mathbb{E}_2[E] = -4\sqrt{2} \approx -5.6585$, 
 * $\log Z_1 = -2$, &nbsp;&nbsp; $\log Z_2 = -4$, &nbsp;&nbsp; $\log Z_2/Z_1 = 2$.

Now, let us say that we obtained random samples from each of the two temperatures, with $N_1 = 10000$ samples from system 1 and $N_2 = 5$ samples from system 2.
The results from one random realization are shown here.

Without UWHAM, we can get the average $E$ for each set of samples, giving:
 * $\mathbb{E}_1[E] = -3.99153$, &nbsp;&nbsp;&nbsp; $\mathbb{E}_2[E] = -6.92858$

Note how the estimate for $\mathbb{E}_2[E]$ is quite poor compared to the exact result, as it is based on very few samples.

In a run of UWHAM on these samples we get:
 * $F_1 - F_2 =1.99595$
 * $\mathbb{E}_1[E] = -3.99218$, &nbsp;&nbsp; $\mathbb{E}_2[E] = -5.63335$
 * $N^\text{eff}_1 =  10004.99$, &nbsp;&nbsp; $N^\text{eff}_2 = 5191.43$

The values $F_1$ and $F_2$ returned by UWHAM are ambigious by a common constant, but their difference should approximate $\log Z_2/Z_1$, which it does: It is only 
-0.00405404 off from the analytic result of 2.

Without UWHAM we only have 5 samples from system 2, but with UWHAM, in a sense we have 5191.4 effective samples for system 2. This suggests why the estimate for $\mathbb{E}_2[E]$ is much more accurate using UWHAM, compared to the analytic result.
