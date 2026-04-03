# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a statistical research project demonstrating phylogenetic comparative analysis using Ornstein-Uhlenbeck (OU) processes for logistic regression with phylogenetic random effects. It uses linguistic data (Grambank features + EDGE language tree) to illustrate Bayesian phylogenetic methods. The published output is at https://profgerhard.de/OU_logistic_example/phylogenetic_OU_regression.html

## Environment Setup

```bash
conda env create -f OU_logistic_example.yml
conda activate OU_logistic_example
```

Julia dependencies (first time):
```bash
cd code/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Notebooks

Render a Quarto notebook to HTML:
```bash
quarto render code/phylogenetic_OU_regression.qmd
quarto render code/phylogenetic_OU_regression_julia.qmd
```

Run interactively: open `.qmd` files in VSCode/RStudio, or convert to Jupyter notebooks. Cached Stan fits are stored as `.rds` files in `code/` to avoid re-running MCMC.

## Architecture

The project has three main analysis notebooks in `code/` and three Stan model files:

**Notebooks (Quarto `.qmd`)**:
- `phylogenetic_OU_regression.qmd` — Main R+Stan analysis. Downloads data, fits three models (vanilla logistic → Brownian random intercept → OU random intercept), compares via bridge sampling and Bayes factors.
- `phylogenetic_OU_regression_julia.qmd` — Julia+Turing equivalent that additionally marginalizes over 902 posterior tree samples from `data/all_trees/`.
- `bayes_in_julia.qmd` — Introductory Julia/Turing tutorial used as scaffolding.

**Stan models** (`code/*.stan`):
- `OU_correlation.stan` — Core model. Implements OU-process covariance, Cholesky-factored correlation matrix, binary logistic likelihood, prior and posterior predictive checks.
- `ctmc_dependent.stan` / `ctmc_independent.stan` — CTMC (continuous-time Markov chain) models for discrete character evolution, used for Bayes factor comparison.

**Julia helper**:
- `bridge_sampling.jl` — Custom bridge sampling implementation for marginal likelihood estimation / Bayes factors (parallelized via `@threads`).

**Data** (`data/`):
- `global-language-tree-MCC-labelled.tree` — EDGE MCC phylogenetic tree (~20 MB).
- `all_trees/` — 902 posterior tree samples for phylogenetic uncertainty analysis.
- `grambank_vals.csv` — Full Grambank linguistic features (~50 MB); `grambank_vals_pruned.csv` is the 100-language subset used in analysis.

Data is downloaded automatically on first notebook run.

## Key Modeling Concepts

- **OU vs. Brownian motion**: OU adds a mean-reversion (stabilizing selection) parameter `alpha`; Brownian motion is the `alpha → 0` limit. OU is preferred for slowly-diverging traits.
- **Phylogenetic non-independence**: Languages in the same family are correlated; the covariance matrix is derived from the phylogenetic tree distance matrix.
- **Model comparison**: Bridge sampling estimates marginal likelihoods; Bayes factors compare competing evolutionary models.
- **Multi-tree marginalization**: The Julia version averages over tree uncertainty by running inference on each of the 902 posterior trees.
