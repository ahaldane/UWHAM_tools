#!/usr/bin/env python
import numpy as np
from numpy import logaddexp
from scipy.special import logsumexp
import argparse, sys, math, textwrap

exps = lambda x: np.exp(x - np.max(x, axis=-1)[...,None])

def UWHAM(BE, nsamples, tol=1e-5, niter=None):
    """
    UWHAM for an arbitrary set of hamiltonians and samples, given the
    B*E value (negative log likelihood) evaluated for each sample for each
    hamiltonian.

    Inputs:
        BE : Matrix of Boltzmann factors (negative log likelihood) of shape
             (N,M) where N is the number of Hamiltonians and M is the total
             number of samples from all Hamiltonians combined. Element (i,j)
             is the Boltzmann factor under Hamiltonian i for sample j.
        nsamples : list of the number of samples from each Hamiltonian
        tol : change in Fs at which to stop iteration
        niter : if given, ignore tol and iterate this many times
    Outputs:
        F : estimated relative free energy for each hamiltonian
        Si : sample-entropy for all samples concatenated together
        w : UWHAM weights for all samples in all Hamiltonians of same shape as
            BE, normalized so the top weight is 1. Use for weighted averages:
              >>> Fs, Si, w = UWHAM(BE, nsamples)
              >>> E1 = np.concatenate([E_S1H1, E_S2H1])
              >>> Ebar0 = np.average(E1, weights=w[0])
    """
    # guess initial entropies and free energies
    Si = np.zeros(np.sum(nsamples), dtype='double')
    Fs = np.zeros(BE.shape[0])

    logN = np.log(nsamples)

    # iterate UWHAM equations
    is_not_finished = get_iteration_condition(niter, tol)
    while is_not_finished(Fs, Si):
        Si = -logsumexp((logN + Fs) - BE.T, axis=1)
        Fs = -logsumexp(Si - BE, axis=1)

    # calculate final UWHAM weights, normalized so top weight is 1
    weights = exps(Si - BE)
    
    return Fs, Si, weights

def nlogdotexp(nM, v):
    # efficiently implements -np.log(np.dot(np.exp(-M), np.exp(v)))
    vx = np.max(v)
    Mx = np.min(nM, axis=-1)

    v0 = v - vx
    M0 = Mx[:,None] - nM

    np.exp(v0, out=v0)
    np.exp(M0, out=M0)

    res = np.dot(M0, v0)
    np.log(res, out=res)
    np.negative(res, out=res)
    res -= vx - Mx
    return res

def UWHAM_manyPT(B, E, tol=1e-6, niter=None, Bt=None):
    """
    UWHAM for the same Hamiltonian at a large number of different temperatures,
    given the E and B of each sample.

    Inputs:
        B : inverse temperatures used to generate each sample
        E : Hamiltonian (energy) of each sample

        tol : change in Fs at which to stop iteration
        niter : if given, ignore tol and iterate this many times
    Outputs:
        F : Free energy for each temperature
        Si : sample-entropy for each sample (for use in computing weights)
        w : UWHAM weights for use in averages
    """
    # This implementation requires N**2 space. Can be modified to require
    # N space if needed, but requires extra mutliply operation in loop
    BE = np.outer(beta, E)
    BEt = BE.T.copy()

    Si = np.zeros(N, dtype='double') # guess initial weights
    Fs = np.zeros(N, dtype='double') # guess initial weights

    is_not_finished = get_iteration_condition(niter, tol)

    # iterate UWHAM equations with logsumexp trick (nlogdotexp)
    while is_not_finished(Fs, Si):
        Si = nlogdotexp(BEt, Fs)
        Fs = nlogdotexp(BE, Si)
    Si = nlogdotexp(BEt, Fs)

    return Fs, Si

def manyPT_weights(Bt, Si, E):
    """
    UWHAM weights for a target temperature for a manyPT dataset

    Inputs:
        Bt : target temperature to compute weights for.
        Si : sample-entropy for each sample
        E : Hamiltonian (energy) of each sample
    Outputs:
        weights : sample weights

    Using the weights one can compute weighted averages:
        >>> Ebar_t = np.average(E, weights=w)
    """
    return exps(Si - Bt*E)

def Neff(weights):
    """
    Effective number of samples corresponding to this set of weights.
    
    This is derived from a weighted average of N bernoulli trials, eg X = (w1
    X1 + w2 X2 + ...)/sum(w) for which we find var(X) = p(1-p)/Neff, which is
    the same functional form as the variance for an unweighted average of Neff
    trials. As expected, we get Neff=N in the unweighted case, Neff=1 in the
    case one weight is very large.

    Inputs:
        weights : sample weights, only last axis is evaluated if dims > 1
    Outputs:
        Neff : estimate of effective number of samples
    """
    return (np.sum(weights, axis=-1)**2)/np.sum(weights**2, axis=-1)

def overlap(weights):
    """
    Measure of the amount of "overlap" of different Hamiltonians, given the
    UWHAM weight matrix.
    
    This uses the measure of overlap provided in eq 10b of 
    C. H. Bennett, J. Comput. Phys. 22, 245 (1976).
    which shows this measure reflects the statistical error in UWHAM
    estimates of relative partition function values.

    Inputs:
        weights : sample weights of shape (N,M) for N Hamiltonians, M samples.
    Outputs:
        ovlap : overlap matrix of shape (N,N) showing the state-space overlap
                of all pairs of Hamiltonians
    """
    # get normalized weights
    wn = weights/np.sum(weights, axis=1, keepdims=True)
    # get "overlap" between hamiltonian state spaces
    from scipy.spatial.distance import pdist, squareform
    return 1-squareform(1-2*pdist(wn, lambda a,b: np.sum(a*b/(a+b))))

def BoltzmannBlock(BE):
    """
    Convert a list-of-lists of Boltzmann factor arrays for UWHAM to a block
    matrix format.

    Inputs:
        BE : list of len N, of lists of len N, of form:
            [[BEb(sa) for sa in samples] for BEb in Hamiltonians]
            where sa is an array of sample coordinates from hamiltonian a and
            BEb is the Boltzmann factor function for hamiltonian b, i.e. the
            inverse temperature B=1/kT times the energy. BEb is also known as
            the negative log likelihood. The number of hamiltonians, and number
            of sample arrays, is N.
    Outputs:
        BEm : Matrix of shape (N,M) for number of Hamiltonians N and total
              number of samples M, giving Boltzmann factor for all combinations.
        nsamples : Number of samples from each Hamiltonian.
    """
    n_hamiltonians = len(BE)
    N_samples = np.array([len(BEij) for BEij in BE[0]])
    if not all(len(BEi) == n_hamiltonians for BEi in BE):
        raise ValueError("Number of sample BE arrays per Hamiltonian differs")
    if not all(len(BEij) == ni for BEi in BE for BEij,ni in zip(BEi,N_samples)):
        raise ValueError("Number of samples differs between hamiltonians")

    # combined BE matrix: axis-0 iterates hamiltonians, axis-1 samples
    BE = np.block(BE)
    return BE, N_samples

def EvaluateHamiltonians(samples, hamiltonians):
    """
    Gets the Boltzmann factor matrix given the callable negative-log-likelihood
    hamiltonian functions and a set of samples from each likelihood function.

    Inputs:
        samples : list of sample-coordinate-vectors, one from each hamiltonian
        hamiltonians: list of callable vectorized functions B*E
    Outputs:
        BEm : Matrix of shape (N,M) for number of Hamiltonians N and total
              number of samples M, giving Boltzmann factor for all combinations.
        nsamples : Number of samples from each Hamiltonian.
    """
    return BoltzmannBlock([[BEb(sa) for sa in samples] for BEb in hamiltonians])

def get_iteration_condition(niter, tol):
    """
    Returns a termination condition function to determine when to stop UWHAM
    iteration, either niter-based or tol-based.
    
    If 'niter' is given, loop for this number of iterations.
    Or 'tol' is given, loop until change in F goes below the given tolerance.
    """
    if niter is not None:
        # niter-based: run for niter loops
        def terminate_condition(F, Si):
           nonlocal niter
           niter -= 1
           return niter != 0
    else:
        # tol-based: run until change in F goes below given tolerance val
        lastdF = np.inf
        def terminate_condition(F, Si):
            F = np.asarray(F)
            nonlocal lastdF
            dF = F - np.mean(F)
            if np.max(np.abs(dF - lastdF)) < tol:
                return False
            lastdF = dF
            return True

    return terminate_condition

def main():
    parser = argparse.ArgumentParser(description='Perform UWHAM Analysis', 
                                     epilog=textwrap.dedent("""
    -------------------------------------------------------------------------

    If `--nsamples` is not given, "BE" should be a set of numpy ".npy" array
    files of negative log-likelihoods, in order:

        BE_S1H1 BE_S2H1 BE_S3H1 ...  BE_S1H2 BE_S2H2 BE_S3H2 ...

    where BE_SiHj is the array of negative log-likelihoods for the sample-set
    generated by potential $i$ evaluated using potential $j$. In a common
    scenario in physics, the negative log-likelihood is equal to the Boltzmann
    factor $B_j E_j(x)$ for inverse temperature $B_j$ and Hamiltonian "energy"
    $E_j(x)$ for sample $x$. The number of "BE" input arrays should be a
    perfect square $N^2$ as there should be one set of samples from each of $N$
    potentials, each evaluated for each of the $N$ potentials.

    If `--nsamples` is supplied it should be either: 1. a `.npy` file of an
    array of the number of samples obtained for each hamiltonian, or 2. a
    comma-separated list of number of samples for each Hamiltonian. If
    `--nsamples` is supplied the `BE` argument should be a single `.npy` file
    with an array of shape `(N,M)` for number of Hamiltonians $N$ and total
    number of samples $M$, such that element `(i,j)` is the Boltzmann factor
    under Hamiltonian $i$ for sample $j$.

    The script will print out:

        "Ns = ..."
            The number of input samples from each Hamiltonian for reference 
        "Neff = ..."
            The effective number of samples corresponding to the computed UWHAM
            weights for each Hamiltonian 
        "ΔlogZ = ..."
            The relative log partition functions of each Hamiltonian, relative
            to the mean. (i.e, relative free energies of each Hamiltonian)
        "Overlap matrix: ..."
            The overlap of state space between all pairs of Hamiltonians.

    If an output file is supplied with the --weights option, the UWHAM weights
    for all samples in all Hamiltonians will be saved to file. This will be an
    array of shape (N,M) for N Hamiltonians, and M total samples which are the
    input samples concatenated together. Use this for weighted averages:

        >>> weights = np.load('weights.npy')
        >>> Vbar1 = np.average(E1, weights=w[0]) # average under 0th potential
    """), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('BE', nargs='*', help="Boltzmann factor files")
    parser.add_argument('--weights', help="save UWHAM weights to this file")
    parser.add_argument('--tol', type=float, default=1e-6,
                        help="change in Fs at which to stop iteration")
    parser.add_argument('--niter', type=int,
                        help="If given, ignore tol and iterate this many times")
    parser.add_argument('--nsamples', help="number of samples per Hamiltonian")

    args = parser.parse_args(sys.argv[1:])

    if args.nsamples is None:
        N = math.isqrt(len(args.BE))
        if N == 0:
            parser.print_help()
            return
        if N*N != len(args.BE):
            raise ValueError("BE arrays must form a square matrix."
                 f"Got {len(args.BE)} arguments which is not a perfect square")

        BE = [np.load(fn) for fn in args.BE]
        BE, nsamples = BoltzmannBlock([BE[i*N:(i+1)*N] for i in range(N)])
    else:
        # try to load a comma-separated set of ints, otherwise load from file
        try:
            nsamples = np.array([int(x) for x in nsamples.split(',')])
        except:
            nsamples = np.load(args.nsamples)

        if len(args.BE) != 1:
            raise ValueError("If nsamples is given, a signle BE matrix should "

                             "be supplied")
        BE = np.load(args.BE[0])
        if np.sum(nsamples) != BE.shape[1]:
            raise ValueError(f"Total number of samples ({np.sum(nsamples)} "
                            f"does not match that in BE arrayy ({BE.shape[1]})")

    tol, niter = args.tol, args.niter

    Fs, Si, weights = UWHAM(BE, nsamples, tol, niter)
    
    print("Ns: " + ", ".join(str(n) for n in nsamples))
    print(f"Neffs: " + ", ".join(f"{n:.3f}" for n in Neff(weights)))
    print(f"ΔlogZ: " + ", ".join(f"{d:.3f}" for d in Fs - np.mean(Fs)))

    print("")
    print("Overlap matrix:")
    print(np.array2string(overlap(weights), precision=3, suppress_small=True,  
                          floatmode='maxprec_equal'))

    if args.weights:
        np.save(weights, args.weights)

def test():
    # Demo system:
    # Two potentials which are same boltzmann distribution at different temps
    beta1, beta2 = 1.0, np.sqrt(2)
    # Density of states is gaussian # with std = s
    s = 2.0
    
    # generate energies from both temperatures
    from scipy.stats import norm
    E1 = norm.rvs(loc=-beta1*s*s, scale=s, size=10000)  
    E2 = norm.rvs(loc=-beta2*s*s, scale=s, size=5)  
    E = np.concatenate([E1, E2])

    # analytical result for F:
    # partition function for gaussian density of states is exp(-s**2 * b**2 / 2)
    # where s is the gaussian std, b is beta (inverse temp).
    Fan1 = -s*s*beta1*beta1/2
    Fan2 = -s*s*beta2*beta2/2
    # the mean energies are \int e^{-BE} Norm(E,0,s) E dE
    Ebar1an = -beta1*s*s
    Ebar2an = -beta2*s*s
    print("Analytic:")
    print(f"F1 = {Fan1:.6g}   F2 = {Fan2:.6g}, diff={Fan1-Fan2:.6g}")
    print(f"N1: {len(E1)}  N2: {len(E2)}")
    print(f"Ebar1: {Ebar1an:.6g}  Ebar2: {Ebar2an:.6g}")

    import pylab as plt
    plt.hist(E1, bins=100, density=True)
    plt.hist(E2, bins=100, density=True)
    plt.show()

    BE_S1H1 = beta1*E1
    BE_S1H2 = beta2*E1
    BE_S2H1 = beta1*E2
    BE_S2H2 = beta2*E2

    BE, nsamples = BoltzmannBlock([[BE_S1H1, BE_S2H1], [BE_S1H2, BE_S2H2]])
    #np.save('BE', BE)
    #np.save('nsamples', nsamples)

    (F1, F2), Si, w = UWHAM(BE, nsamples)
    Ebar1, Ebar2 = np.average(E, weights=w[0]), np.average(E, weights=w[1])
    ne = Neff(w)

    print("")
    print("Computed:")
    print(f"F1 = {F1:.6g}   F2 = {F2:.6g}, diff={F1-F2:.6g}")
    print(f"Neff1: {ne[0]:.9g}  Neff2: {ne[1]:.7g}")
    print(f"Ebar1: {Ebar1:.6g}  Ebar2: {Ebar2:.6g}")
    #print("w1:", np.array2string(w1, edgeitems=2))
    #print("Si:", np.array2string(Si, edgeitems=2))
    print("")
    print(f"Accuracy in Delta F: {F1-F2:.6g} vs {Fan1-Fan2:.6g}, "
          f"difference = {(F1-F2)-(Fan1-Fan2):.6g}")
    
    print("")
    print("Overlap matrix:")
    print(overlap(w))

    print("")
    print("Without UWHAM: (simple average)")
    print(f"Ebar1: {np.mean(E1):.6g}  Ebar2: {np.mean(E2):.6g}")


    Es = np.concatenate([E1, E2])
    plt.plot(Es, Si, '.')
    plt.plot(Es, -Es**2/(2*s*s), '.')
    plt.xlim(-20,0)
    plt.ylim(-50,10)

    plt.figure()
    plt.plot(Es, w[0], '.')
    plt.plot(Es, w[1], '.')
    plt.show()

    print("")
    print("Test with a single Hamiltonian")
    Fs, Si, w = UWHAM(BE_S1H1[None,:], [len(BE_S1H1)])
    print("F", Fs)
    print("w:", np.array2string(w[0], edgeitems=2, threshold=5))
    print("Neff:", Neff(w[0]))

def UWHAM_krylov(BE, nsamples, tol=1e-5, niter=None):
    """
    Same as UWHAM, but uses a krylov solver.
    """
    Nf, Ns = BE.shape[0], np.sum(nsamples)
    IF, IS = np.eye(Nf), np.eye(Ns)

    logN = np.log(nsamples)

    def fun(x):
        Fs, Si = x[:Nf], x[Nf:]

        eF = Si - BE
        eS = (logN + Fs) - BE.T

        f = np.concatenate([Fs + logsumexp(eF, axis=1), 
                            Si + logsumexp(eS, axis=1)])
        return f
        #jac = np.block([[             IF, np.exp(eF + Fs[:,None])],
        #                [np.exp(eS + Si[:,None]),              IS]])
        #print(jac.shape)
        #return f, jac

    x0 = np.zeros(Nf + Ns, dtype='f8')
    sol = scipy.optimize.root(fun, x0, method='krylov')
    Fs, Si = sol.x[:Nf], sol.x[Nf:]

    # calculate final UWHAM weights, normalized so top weight is 1
    weights = exps(Si - BE)
    
    return Fs, Si, weights

if __name__ == '__main__':
    main()
    #test()
