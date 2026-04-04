
# Extract only stochastic (non-deterministic) parameter samples from a chain,
# using the LogDensityFunction's VarInfo to exclude := variables.
function stochastic_samples(chain, ldf)
    vi_syms = Set(DynamicPPL.getsym(k) for k in keys(ldf.varinfo))
    param_names = filter(names(chain, :parameters)) do nm
        Symbol(split(string(nm), "[")[1]) in vi_syms
    end
    Array(chain[:, param_names, :])
end


# Warp-3 bridge sampling (Meng & Schilling 2002).
#
# The proposal in the transformed space is a standard normal, which is
# equivalent to drawing each z_d independently from TDist(t_df) and then
# mapping θ = L * z + μ (where L is the Cholesky factor of the posterior
# sample covariance).  The proposal density cancels from the bridge ratio:
#
#   l(θ) = log p*(θ) + log|det L| − Σ_d log pdf(TDist(t_df), z_d)
#
# This is far better calibrated than a plain Gaussian proposal for the
# non-Gaussian, high-dimensional posteriors arising in phylogenetic models.
function bridge_sampling(
    samples::Array{Float64,2},
    log_density::Function;
    verbose   = false,
    t_df      = 3.0,        # degrees of freedom for warp-3 t-proposal
    n_prop_mult = 1,        # proposal samples = n_prop_mult * n_post
)
    n  = size(samples, 1)
    d  = size(samples, 2)
    nr = n ÷ 2

    samples_4_fit  = samples[1:nr, :]        # (nr, d) — fit the warp
    samples_4_iter = samples[(nr+1):end, :]  # (n-nr, d) — iterate

    n_post = size(samples_4_iter, 1)
    n_prop = n_post * n_prop_mult

    # --- fit warp ---
    μ = vec(mean(samples_4_fit, dims = 1))
    Σ = Symmetric(cov(samples_4_fit) + 1e-8 * I(d))
    C = cholesky(Σ)
    L = C.L
    logdetL = logdet(L)

    t = TDist(t_df)

    # Warp-3 log-ratio  l(θ) = log p*(θ) + log|det L| − Σ_d log pdf(t, z_d)
    function warp3_l(theta::AbstractVector)
        z     = L \ (theta .- μ)
        log_density(theta) + logdetL - sum(logpdf.(t, z))
    end

    # --- l1: posterior samples in warp-3 space ---
    l1 = Vector{Float64}(undef, n_post)
    for i in 1:n_post
        l1[i] = warp3_l(samples_4_iter[i, :])
    end

    # --- l2: proposal samples (z_d iid ~ TDist(t_df), θ = L z + μ) ---
    l2 = Vector{Float64}(undef, n_prop)
    for j in 1:n_prop
        z_prop     = rand(t, d)
        theta_prop = L * z_prop .+ μ
        l2[j]      = warp3_l(theta_prop)
    end

    # --- bridge sampling fixed-point iteration ---
    lstar = median(l1)
    n_1   = length(l1)
    n_2   = length(l2)
    neff  = Float64(n_1)
    s1    = neff / (neff + n_2)
    s2    = n_2  / (neff + n_2)

    function upd(logr)
        lognumi = logsumexp([
            l2[i] - lstar - logaddexp(log(s1) + l2[i] - lstar, log(s2) + logr)
            for i in 1:n_2
        ])
        logdeni = logsumexp([
            -logaddexp(log(s1) + l1[i] - lstar, log(s2) + logr)
            for i in 1:n_1
        ])
        log(n_1) - log(n_2) + lognumi - logdeni
    end

    if verbose
        println("  dim=$(d), n_post=$(n_post), n_prop=$(n_prop)")
        println("  l1: mean=$(round(mean(l1),digits=2))  std=$(round(std(l1),digits=2))  min=$(round(minimum(l1),digits=2))  max=$(round(maximum(l1),digits=2))")
        println("  l2: mean=$(round(mean(l2),digits=2))  std=$(round(std(l2),digits=2))  min=$(round(minimum(l2),digits=2))  max=$(round(maximum(l2),digits=2))")
        println("  lstar=$(round(lstar,digits=2))  logdetL=$(round(logdetL,digits=2))")
    end

    ft = optimize(y -> (upd(y) - y)^2, -1e6, 1e6)
    ft.minimizer + lstar
end
