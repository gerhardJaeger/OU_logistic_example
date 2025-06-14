data {
  int<lower=1> N;                         // number of tips with data
  int<lower=1> Nnodes;                    // total number of nodes
  int<lower=1> Nedges;                    // number of edges in the tree
  array[N] int<lower=0, upper=1> x;       // first binary response
  array[N] int<lower=0, upper=1> y;       // second binary response
  array[Nedges, 2] int<lower=1, upper=Nnodes> edges;     // parent → child
  vector<lower=0>[Nedges] edge_lengths;   // edge lengths
  int<lower=1, upper=Nnodes> root_node;   // index of root node
}

parameters {
  matrix[Nnodes, 2] z_std;                   // standard-normal innovations
  vector<lower=0>[2] sigma;                  // OU diffusion parameters
  vector<lower=0>[2] lambda;                 // OU pull strength
  vector[2] mu;                              // OU stationary means
  cholesky_factor_corr[2] L_std;             // Cholesky factor of correlation matrix
}

transformed parameters {
  matrix[Nnodes, 2] z;  // latent values
  real rho = L_std[2, 1];  // correlation coefficient

  // Root node
  z[root_node] = (mu + (sigma ./ sqrt(2 * lambda)) .* (L_std * to_vector(z_std[root_node])))';



  // Recursive evolution
  for (e in 1:Nedges) {
    int edge_index = Nedges - e + 1;
    int parent = edges[edge_index, 1];
    int child  = edges[edge_index, 2];
    real len = edge_lengths[edge_index];

    // Vectorized decay and scale
    vector[2] decay = exp(-lambda * len);
    vector[2] s = sigma .* sqrt(-expm1(-2 * lambda * len) ./ (2 * lambda));
    vector[2] mean = mu + (to_vector(z[parent]) - mu) .* decay;
    vector[2] eps  = L_std * z_std[child]';
    z[child] = (mean + s .* eps)';  // now row_vector[2]
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
    x[i] ~ bernoulli_logit(z[i, 1]);
    y[i] ~ bernoulli_logit(z[i, 2]);
  }
}
