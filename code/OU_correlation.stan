data {
  int<lower=1> N;                                        // number of tips with data
  int<lower=1> Nnodes;                                   // total number of nodes
  int<lower=1> Nedges;                                   // number of edges in the tree
  array[N] int<lower=0, upper=1> x;                      // first binary response
  array[N] int<lower=0, upper=1> y;                      // second binary response
  array[Nedges, 2] int<lower=1, upper=Nnodes> edges;     // parent → child
  vector<lower=0>[Nedges] edge_lengths;                  // edge lengths
  int<lower=1, upper=Nnodes> root_node;                  // index of root node
}

parameters {
  matrix[Nnodes, 2] z_std;                   // standard-normal latent variables
  vector<lower=0>[2] sigma;                  // OU diffusion parameters
  vector<lower=0>[2] lambda;                 // OU pull strengths
  vector[2] mu;                              // OU stationary means
  cholesky_factor_corr[2] L_std;             // Cholesky factor of correlation matrix
}

transformed parameters {
  matrix[Nnodes, 2] z;     // latent values
  real rho = L_std[2, 1];  // correlation coefficient

  // Root node, drawn from OU stationary distribution
  z[root_node] = (mu + (sigma ./ sqrt(2 * lambda)) .* (L_std * to_vector(z_std[root_node])))';



  // Recursive evolution
  for (e in 1:Nedges) {
    int edge_index = Nedges - e + 1;   // reverse order for recursion, root to tips
    int parent = edges[edge_index, 1];
    int child  = edges[edge_index, 2];
    real len = edge_lengths[edge_index];

    // Vectorized decay and scale
    vector[2] decay = exp(-lambda * len);
    vector[2] s = sigma .* sqrt(-expm1(-2 * lambda * len) ./ (2 * lambda));
    vector[2] mean = mu + (to_vector(z[parent]) - mu) .* decay;
    vector[2] eps  = L_std * z_std[child]';
    z[child] = (mean + s .* eps)';
  }
}

model {
  // Priors
  sigma ~ lognormal(0, 1);
  lambda ~ lognormal(0, 1);
  mu ~ normal(0, 2);
  L_std ~ lkj_corr_cholesky(2.0);
  to_vector(z_std) ~ normal(0, 1);

  // Likelihood
  for (i in 1:N) {
    # x[i] ~ bernoulli_logit(z[i, 1]);
    target += bernoulli_logit_lpmf(x[i] | z[i, 1]);
    # y[i] ~ bernoulli_logit(z[i, 2]);
    target += bernoulli_logit_lpmf(y[i] | z[i, 2]);
  }
}


generated quantities {
  vector[N] log_lik_x;
  vector[N] log_lik_y;
  array[N] int x_rep;
  array[N] int y_rep;
  array[N] int x_prior;
  array[N] int y_prior;

  // Posterior predictive
  for (i in 1:N) {
    x_rep[i] = bernoulli_logit_rng(z[i, 1]);
    y_rep[i] = bernoulli_logit_rng(z[i, 2]);
    log_lik_x[i] = bernoulli_logit_lpmf(x[i] | z[i, 1]);
    log_lik_y[i] = bernoulli_logit_lpmf(y[i] | z[i, 2]);
  }

  // Prior predictive simulation
  {
    vector[2] mu_prior;
    vector[2] sigma_prior;
    vector[2] lambda_prior;
    matrix[2, 2] L_prior;
    matrix[Nnodes, 2] z_std_prior;
    matrix[Nnodes, 2] z_prior;

    for (j in 1:2) {
      mu_prior[j] = normal_rng(0.0, 2.0);
      sigma_prior[j] = lognormal_rng(0.0, 1.0);
      lambda_prior[j] = lognormal_rng(0.0, 1.0);
    }
    L_prior = cholesky_decompose(lkj_corr_rng(2, 2.0));

    for (i in 1:Nnodes)
      z_std_prior[i] = to_row_vector(normal_rng(rep_vector(0.0, 2), rep_vector(1.0, 2)));

    // Root node
    z_prior[root_node] = (mu_prior + (sigma_prior ./ sqrt(2 * lambda_prior)) .* (L_prior * to_vector(z_std_prior[root_node])))';

    // Recursion
    for (e in 1:Nedges) {
      int edge_index = Nedges - e + 1;
      int parent = edges[edge_index, 1];
      int child  = edges[edge_index, 2];
      real len = edge_lengths[edge_index];

      vector[2] decay = exp(-lambda_prior * len);
      vector[2] s = sigma_prior .* sqrt(-expm1(-2 * lambda_prior * len) ./ (2 * lambda_prior));
      vector[2] mean = mu_prior + (to_vector(z_prior[parent]) - mu_prior) .* decay;
      vector[2] eps = L_prior * z_std_prior[child]';
      z_prior[child] = (mean + s .* eps)';
    }

    for (i in 1:N) {
      x_prior[i] = bernoulli_logit_rng(z_prior[i, 1]);
      y_prior[i] = bernoulli_logit_rng(z_prior[i, 2]);
    }
  }

  // Simulate using posterior parameters but fresh noise z_std_prior
  array[N] int x_replicated_noise;
  array[N] int y_replicated_noise;
  {
    matrix[Nnodes, 2] z_std_resample;
    matrix[Nnodes, 2] z_resample;

    for (i in 1:Nnodes)
      z_std_resample[i] = to_row_vector(normal_rng(rep_vector(0.0, 2), rep_vector(1.0, 2)));

    // Root node
    z_resample[root_node] = (mu + (sigma ./ sqrt(2 * lambda)) .* (L_std * to_vector(z_std_resample[root_node])))';

    // Recursion
    for (e in 1:Nedges) {
      int edge_index = Nedges - e + 1;
      int parent = edges[edge_index, 1];
      int child  = edges[edge_index, 2];
      real len = edge_lengths[edge_index];

      vector[2] decay = exp(-lambda * len);
      vector[2] s = sigma .* sqrt(-expm1(-2 * lambda * len) ./ (2 * lambda));
      vector[2] mean = mu + (to_vector(z_resample[parent]) - mu) .* decay;
      vector[2] eps = L_std * z_std_resample[child]';
      z_resample[child] = (mean + s .* eps)';
    }

    for (i in 1:N) {
      x_replicated_noise[i] = bernoulli_logit_rng(z_resample[i, 1]);
      y_replicated_noise[i] = bernoulli_logit_rng(z_resample[i, 2]);
    }
  }

}

