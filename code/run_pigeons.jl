
# Standalone script: compute log marginal likelihoods via Pigeons.jl
# stepping-stone sampling for all 8 models.
# Results saved to cache/pigeons_*.jld2.
# Run from the code/ directory:
#   julia --project=. run_pigeons.jl

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Distributions
using DynamicPPL
using ExponentialUtilities
using LinearAlgebra
using LogDensityProblems
using Phylo
using Random
using RCall
using Statistics
using StatsBase
using StatsFuns
using JLD2
using Turing
using Pigeons

include("bridge_sampling.jl")

# ─── Data loading ────────────────────────────────────────────────────────────

isdir("../data") || mkpath("../data")
tree_gz   = "../data/global-language-tree-MCC-labelled.tree.gz"
tree_file = "../data/global-language-tree-MCC-labelled.tree"

if !isfile(tree_file)
    Downloads.download(
        "https://github.com/rbouckaert/global-language-tree-pipeline/releases/download/v1.0.0/global-language-tree-MCC-labelled.tree.gz",
        tree_gz
    )
    open(tree_gz) do f_in
        open(tree_file, "w") do f_out
            write(f_out, transcode(GzipDecompressor, read(f_in)))
        end
    end
    rm(tree_gz)
end

R"""
library(ape)
.full_tree <- read.nexus($tree_file)
.full_tree$tip.label <- sapply(strsplit(.full_tree$tip.label, "_"), `[`, 1)
.full_tree$edge.length <- .full_tree$edge.length / mean(.full_tree$edge.length)
"""

# Load pruned dataset directly — it was created by the notebook and already has x, y columns.
pruned_csv = "../data/grambank_vals_pruned.csv"
isfile(pruned_csv) || error("Run the notebook first to create $pruned_csv")
d = CSV.read(pruned_csv, DataFrame)

keep_langs = d.Language_ID
R"""
.pruned <- drop.tip(.full_tree, setdiff(.full_tree$tip.label, $keep_langs))
.pruned <- reorder(.pruned, "postorder")
.tip_labels  <- .pruned$tip.label
.newick_str  <- write.tree(.pruned)
"""

taxa_ordered = rcopy(R"as.character(.tip_labels)")
newick_str   = rcopy(R".newick_str")
edge_tree    = parsenewick(newick_str)

# Reorder d to match the tip order in the pruned tree
idx  = indexin(taxa_ordered, d.Language_ID)
d    = d[idx, :]
taxa = taxa_ordered

# ─── Tree helper functions ────────────────────────────────────────────────────

function decompose_tree(tree::RootedTree)
    root = getroot(tree).name
    mothers = String[]; daughters = String[]; lengths = Float64[]
    for nd in traversal(tree, postorder)
        if !isroot(tree, nd)
            mother = getparent(tree, nd)
            br     = getinbound(tree, nd)
            push!(mothers, mother.name)
            push!(daughters, nd.name)
            push!(lengths, getlength(tree, br))
        end
    end
    nodes = sort!(unique(vcat(mothers, daughters)))
    nodes_dict = Dict{String,Int}(nodes .=> 1:length(nodes))
    (root=root, mothers=mothers, daughters=daughters,
     lengths=lengths, nodes_dict=nodes_dict)
end

function get_vcv(tree; taxa=taxa)
    root = getroot(tree)
    tips = getleaves(tree)
    tip_names = getleafnames(tree)
    n = length(tips)
    vcv = zeros(n, n)
    for i in 1:n
        for j in i:n
            if i == j
                vcv[i, j] = distance(tree, root, tips[i])
            else
                vcv[j, i] = vcv[i, j] = distance(tree, root, mrca(tree, [tips[i], tips[j]]))
            end
        end
    end
    return vcv[indexin(tip_names, taxa), indexin(tip_names, taxa)]
end

tree_info  = decompose_tree(edge_tree)
scaled_vcv = get_vcv(edge_tree) ./ 100
scaled_chol = cholesky(scaled_vcv).L
x_centered = d.x .- mean(d.x)

# ─── CTMC helpers ────────────────────────────────────────────────────────────

const N_STATES_CTMC = 4
const N_RATES_INDEP = 4
const N_RATES_DEP   = 8

function rates_to_Q(rates::AbstractVector{T}, n::Int) where T <: Real
    Q = zeros(T, n, n)
    k = 1
    for i in 1:n, j in (i+1):n; Q[i,j] = rates[k]; k += 1; end
    for i in 2:n, j in 1:(i-1); Q[i,j] = rates[k]; k += 1; end
    for i in 1:n; Q[i,i] = -sum(Q[i,:]); end
    return Q
end

function stationary_dist(Q::AbstractMatrix{T}) where T <: Real
    n = size(Q,1)
    A = vcat(Q', ones(T,1,n))
    b = vcat(zeros(T,n), one(T))
    return A \ b
end

function pruning_loglik(s, edges, edge_lengths, root_node, n_nodes, n_tips, Q, stat)
    n_states = size(Q,1)
    L = ones(eltype(Q), n_nodes, n_states)
    for i in 1:n_tips
        for k in 1:n_states; L[i,k] = (k == s[i]) ? one(eltype(Q)) : zero(eltype(Q)); end
    end
    n_edges = size(edges,1)
    for e in 1:n_edges
        parent = edges[e,1]; child = edges[e,2]; len = edge_lengths[e]
        P = exp(Q * len)
        L[parent,:] .*= P * L[child,:]
    end
    return log(dot(stat, L[root_node,:]))
end

function tree_to_edge_arrays(tree::RootedTree, taxa::Vector{String})
    root_name = getroot(tree).name
    n_tips    = length(taxa)
    node_idx  = Dict{String,Int}()
    for (i,t) in enumerate(taxa); node_idx[t] = i; end
    next_idx = n_tips + 1
    for nd in traversal(tree, preorder)
        if !haskey(node_idx, nd.name); node_idx[nd.name] = next_idx; next_idx += 1; end
    end
    root_node = node_idx[root_name]
    n_nodes   = length(node_idx)
    parents = Int[]; children = Int[]; lengths = Float64[]
    for nd in traversal(tree, postorder)
        if !isroot(tree, nd)
            mother = getparent(tree, nd)
            br     = getinbound(tree, nd)
            push!(parents,  node_idx[mother.name])
            push!(children, node_idx[nd.name])
            push!(lengths,  getlength(tree, br))
        end
    end
    (edges=[parents children], lengths=lengths,
     root_node=root_node, n_nodes=n_nodes, n_tips=n_tips, node_idx=node_idx)
end

ctmc_tree      = tree_to_edge_arrays(edge_tree, taxa)
tip_states_obs = 2 .* d.x .+ d.y .+ 1

# ─── Model definitions ───────────────────────────────────────────────────────

@model function vanilla_regression(x_centered, y)
    N = length(y)
    alpha ~ TDist(3) * 2.5
    beta  ~ Normal(0, 2)
    for i in 1:N; y[i] ~ BernoulliLogit(alpha + beta * x_centered[i]); end
end

@model function vanilla_correlation(N)
    mu    ~ MvNormal(zeros(2), 2.0)
    rho_u ~ Normal()
    rho   := 2 * cdf(Normal(), rho_u) - 1
    sigma_u ~ filldist(Normal(0.0, 1.0), 2)
    sigma   := exp.(sigma_u)
    Sigma_l  = [1.0 0.0; rho sqrt(1 - rho^2)]
    z_std ~ filldist(MvNormal(zeros(2), 1.0), N)
    z     := diagm(sigma) * Sigma_l * z_std .+ mu
    x ~ product_distribution(BernoulliLogit.(z[1, :]))
    y ~ product_distribution(BernoulliLogit.(z[2, :]))
end

@model function ctmc_independent(tip_states, ctmc_tree)
    log_r ~ MvNormal(zeros(4), I)
    lr = clamp.(log_r, -8.0, 8.0)
    r  = exp.(lr)
    # independent model: share rates across marginals
    Q = zeros(eltype(r), 4, 4)
    Q[1,2] = r[1]; Q[1,3] = r[2]; Q[2,4] = r[2]; Q[3,4] = r[1]
    Q[2,1] = r[3]; Q[3,1] = r[4]; Q[4,2] = r[4]; Q[4,3] = r[3]
    for i in 1:4; Q[i,i] = -sum(Q[i,:]); end
    stat = stationary_dist(Q)
    Turing.@addlogprob! pruning_loglik(
        tip_states, ctmc_tree.edges, ctmc_tree.lengths,
        ctmc_tree.root_node, ctmc_tree.n_nodes, ctmc_tree.n_tips, Q, stat)
end

@model function ctmc_dependent(tip_states, ctmc_tree)
    log_r ~ MvNormal(zeros(8), I)
    lr    = clamp.(log_r, -8.0, 8.0)
    rates = exp.(lr)
    Q = zeros(eltype(rates), 4, 4)
    Q[1,2] = rates[1]; Q[1,3] = rates[2]; Q[2,4] = rates[3]; Q[3,4] = rates[4]
    Q[2,1] = rates[5]; Q[3,1] = rates[6]; Q[4,2] = rates[7]; Q[4,3] = rates[8]
    for i in 1:4; Q[i,i] = -sum(Q[i,:]); end
    stat = stationary_dist(Q)
    Turing.@addlogprob! pruning_loglik(
        tip_states, ctmc_tree.edges, ctmc_tree.lengths,
        ctmc_tree.root_node, ctmc_tree.n_nodes, ctmc_tree.n_tips, Q, stat)
end

@model function OU_regression(d, tree_info, taxa)
    N = length(taxa)
    root = tree_info.root; mothers = tree_info.mothers
    daughters = tree_info.daughters; lengths = tree_info.lengths
    nodes_dict = tree_info.nodes_dict; n_nodes = length(nodes_dict)
    alpha ~ LocationScale(0.0, 2.5, TDist(3))
    beta  ~ Normal(0, 2); mu ~ Normal(0, 2)
    sigma_u ~ Normal(0, 1); sigma  := exp(sigma_u)
    lambda_u ~ Normal(0, 1); lambda := exp(lambda_u)
    z_std ~ filldist(Normal(), n_nodes)
    z = zeros(typeof(sigma), n_nodes)
    z[nodes_dict[root]] = mu + (sigma / sqrt(2 * lambda)) * z_std[nodes_dict[root]]
    for i in length(mothers):-1:1
        dgt = nodes_dict[daughters[i]]; mth = nodes_dict[mothers[i]]; len = lengths[i]
        decay = exp(-lambda * len)
        s  = sigma * sqrt(-expm1(-2 * lambda * len) / (2 * lambda))
        mn = mu + (z[mth] - mu) * decay
        z[dgt] = mn + s * z_std[dgt]
    end
    x_c = d.x .- mean(d.x)
    for (k, t) in enumerate(taxa)
        d.y[k] ~ BernoulliLogit(alpha + beta * x_c[k] + z[nodes_dict[t]])
    end
end

@model function brownian_regression(d, scaled_chol, taxa)
    N = length(taxa)
    alpha ~ LocationScale(0.0, 2.5, TDist(3))
    beta  ~ Normal(0, 2)
    rate_u ~ Normal(0, 1); rate := exp(rate_u)
    z_std ~ filldist(Normal(), N)
    z := rate .* (scaled_chol * z_std)
    x_c = d.x .- mean(d.x)
    for i in 1:N; d.y[i] ~ BernoulliLogit(alpha + beta * x_c[i] + z[i]); end
end

@model function OU_correlation_single_tree(tree_info, taxa, x, y)
    root = tree_info.root; mothers = tree_info.mothers
    daughters = tree_info.daughters; lengths = tree_info.lengths
    nodes_dict = tree_info.nodes_dict; n_edges = length(mothers); N = length(taxa)
    rho_u ~ Normal(); rho := 2 * cdf(Normal(), rho_u) - 1
    Sigma_l = [1.0 0.0; rho sqrt(1 - rho^2)]
    sigma_u ~ filldist(Normal(), 2); sigma := exp.(sigma_u)
    lambda_u ~ filldist(Normal(), 2); lambda := exp.(lambda_u)
    mu ~ MvNormal(zeros(2), 2.0)
    z_std ~ filldist(MvNormal(zeros(n_edges), ones(n_edges)), 2)
    z_cor = z_std * Sigma_l
    root_states ~ MvNormal(mu, sigma ./ sqrt.(2 .* lambda))
    n_nodes = length(nodes_dict)
    z = zeros(typeof(sigma[1]), n_nodes, 2)
    z[nodes_dict[root], :] .= root_states
    for i in length(mothers):-1:1
        dgt = nodes_dict[daughters[i]]; mth = nodes_dict[mothers[i]]; len = lengths[i]
        decay = exp.(-lambda .* len)
        s  = sigma .* sqrt.(-expm1.(-2 .* lambda .* len) ./ (2 .* lambda))
        mn = mu .+ (z[mth, :] .- mu) .* decay
        z[dgt, :] = mn .+ s .* z_cor[i, :]
    end
    x_logits := [z[nodes_dict[t], 1] for t in taxa]
    y_logits := [z[nodes_dict[t], 2] for t in taxa]
    x ~ product_distribution(BernoulliLogit.(x_logits))
    y ~ product_distribution(BernoulliLogit.(y_logits))
end

@model function brownian_correlation_single(N, scaled_chol)
    rho_u ~ Normal(); rho := 2 * cdf(Normal(), rho_u) - 1
    Sigma_l = [1.0 0.0; rho sqrt(1 - rho^2)]
    rates_u ~ filldist(Normal(), 2); rates := exp.(rates_u)
    mu ~ MvNormal(zeros(2), 2.0)
    z_std ~ filldist(MvNormal(zeros(N), ones(N)), 2)
    z := diagm(rates) * Sigma_l * z_std' * scaled_chol' .+ mu
    x ~ product_distribution(BernoulliLogit.(z[1, :]))
    y ~ product_distribution(BernoulliLogit.(z[2, :]))
end

# ─── Pigeons stepping-stone ───────────────────────────────────────────────────

function pigeons_logml(model, cache_file; n_rounds=12, n_chains=20)
    if isfile(cache_file)
        logml = load(cache_file, "logml")
        println("  [cache] ", cache_file, " -> ", round(logml, digits=3))
        return logml
    end
    println("  Running Pigeons: n_rounds=$(n_rounds), n_chains=$(n_chains)...")
    pt = pigeons(
        target   = TuringLogPotential(model),
        n_rounds = n_rounds,
        n_chains = n_chains,
    )
    logml = stepping_stone(pt)
    mkpath("cache")
    jldsave(cache_file; logml=logml)
    println("  logml = ", logml)
    logml
end

println("\n=== Pigeons stepping-stone log marginal likelihoods ===\n")

println("1. Vanilla regression (small model, n_rounds=10, n_chains=10)")
lg_vr = pigeons_logml(
    vanilla_regression(x_centered, d.y),
    "cache/pigeons_vanilla_regression.jld2"; n_rounds=10, n_chains=10)

println("\n2. Vanilla correlation")
lg_vc = pigeons_logml(
    vanilla_correlation(nrow(d)) | (x=d.x, y=d.y),
    "cache/pigeons_vanilla_correlation.jld2")

println("\n3. CTMC independent (n_rounds=10, n_chains=10)")
lg_ci = pigeons_logml(
    ctmc_independent(tip_states_obs, ctmc_tree),
    "cache/pigeons_ctmc_indep.jld2"; n_rounds=10, n_chains=10)

println("\n4. CTMC dependent (n_rounds=10, n_chains=10)")
lg_cd = pigeons_logml(
    ctmc_dependent(tip_states_obs, ctmc_tree),
    "cache/pigeons_ctmc_dep.jld2"; n_rounds=10, n_chains=10)

println("   Log BF (indep vs dep): ", lg_ci - lg_cd)

println("\n5. Brownian regression")
lg_br = pigeons_logml(
    brownian_regression(d, scaled_chol, taxa),
    "cache/pigeons_brownian_regression.jld2")

println("\n6. OU regression (n_rounds=14)")
lg_or = pigeons_logml(
    OU_regression(d, tree_info, taxa),
    "cache/pigeons_OU_regression.jld2"; n_rounds=14, n_chains=20)

println("   Log BF (OU vs Brownian regression): ", lg_or - lg_br)

println("\n7. Brownian correlation (n_rounds=14)")
lg_bc = pigeons_logml(
    brownian_correlation_single(nrow(d), scaled_chol) | (x=d.x, y=d.y),
    "cache/pigeons_brownian_correlation.jld2"; n_rounds=14, n_chains=20)

println("\n8. OU correlation (n_rounds=14)")
lg_oc = pigeons_logml(
    OU_correlation_single_tree(tree_info, taxa, d.x, d.y),
    "cache/pigeons_OU_correlation.jld2"; n_rounds=14, n_chains=20)

println("   Log BF (OU correlation vs OU regression): ", lg_oc - lg_or)

println("\n=== Summary ===")
println("Vanilla regression:      ", round(lg_vr, digits=3))
println("Vanilla correlation:     ", round(lg_vc, digits=3))
println("CTMC independent:        ", round(lg_ci, digits=3))
println("CTMC dependent:          ", round(lg_cd, digits=3))
println("Brownian regression:     ", round(lg_br, digits=3))
println("Brownian correlation:    ", round(lg_bc, digits=3))
println("OU regression:           ", round(lg_or, digits=3))
println("OU correlation:          ", round(lg_oc, digits=3))
