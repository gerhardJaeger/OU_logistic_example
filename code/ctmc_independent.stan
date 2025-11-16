data {
    int<lower=1> N;
    int<lower=1> Nnodes;
    int<lower=1> Nedges;
    array[N] int<lower=0, upper=1> x;
    array[N] int<lower=0, upper=1> y;
    array[Nedges, 2] int<lower=1, upper=Nnodes> edges;
    vector<lower=0>[Nedges] edge_lengths;
    int<lower=1> root_node;
}

parameters {
    simplex[4] rel_rates;
    real<lower=0> total_rate;
}

transformed parameters {
    matrix[4, 4] Q = rep_matrix(0.0, 4, 4);

    Q[1, 2] = total_rate * rel_rates[1];
    Q[2, 1] = total_rate * rel_rates[2];
    Q[1, 3] = total_rate * rel_rates[3];
    Q[3, 1] = total_rate * rel_rates[4];
    Q[2, 4] = total_rate * rel_rates[3];
    Q[4, 2] = total_rate * rel_rates[4];
    Q[3, 4] = total_rate * rel_rates[1];
    Q[4, 3] = total_rate * rel_rates[2];

    for (i in 1:4) {
        Q[i, i] = -sum(Q[i, ]);
    }

    // stationary-ish distribution via long-time transition
    matrix[4, 4] P = matrix_exp(Q * 1e3);
    vector[4] stat_dist = to_vector(P[1]);
    stat_dist /= sum(stat_dist);

        matrix[Nnodes, 4] loglikelihood;
    loglikelihood = rep_matrix(0.0, Nnodes, 4);

    // tip likelihoods
    for (i in 1:N) {
        int idx = 2 * x[i] + y[i] + 1;
        loglikelihood[i] = rep_row_vector(negative_infinity(), 4);
        loglikelihood[i, idx] = 0;
    }

    // pruning recursion
    for (e in 1:Nedges) {
        int parent = edges[e, 1];
        int child = edges[e, 2];
        real t = edge_lengths[e];
        matrix[4, 4] Plocal = matrix_exp(Q * t);

        for (k in 1:4) {
            loglikelihood[parent, k] += log_sum_exp(
                to_vector(log(Plocal[k]) + loglikelihood[child])
            );
        }
    }
    real ll = log_sum_exp(loglikelihood[root_node] + log(stat_dist)');

}

model {
    // priors
    rel_rates ~ dirichlet(rep_vector(1.0, 4));
    total_rate ~ lognormal(-1, 0.5);

    target += ll;
}

generated quantities {
    real total_rate_prior = lognormal_rng(-1, 0.5);
    vector[4] rel_rates_prior = dirichlet_rng(rep_vector(1.0, 4));
    matrix[4, 4] Qprior = rep_matrix(0.0, 4, 4);

    Qprior[1, 2] = total_rate_prior * rel_rates_prior[1];
    Qprior[2, 1] = total_rate_prior * rel_rates_prior[2];
    Qprior[1, 3] = total_rate_prior * rel_rates_prior[3];
    Qprior[3, 1] = total_rate_prior * rel_rates_prior[4];
    Qprior[2, 4] = total_rate_prior * rel_rates_prior[3];
    Qprior[4, 2] = total_rate_prior * rel_rates_prior[4];
    Qprior[3, 4] = total_rate_prior * rel_rates_prior[1];
    Qprior[4, 3] = total_rate_prior * rel_rates_prior[2];
    
    for (i in 1:4) {
        Qprior[i, i] = -sum(Qprior[i, ]);
    }

    matrix[4, 4] Pprior = matrix_exp(Qprior * 1e3);
    vector[4] stat_dist_prior = to_vector(Pprior[1]);
    stat_dist_prior /= sum(stat_dist_prior);
    
    array[Nnodes] int<lower=1, upper=4> sim_states;
    sim_states[root_node] = categorical_rng(stat_dist_prior);
    
    // Loop backward through post-ordered edges
    for (i in 1:Nedges) {
        int e = Nedges - i + 1;
        int parent = edges[e, 1];
        int child = edges[e, 2];
        real t = edge_lengths[e];
        matrix[4, 4] Plocal = matrix_exp(Qprior * t);
        sim_states[child] = categorical_rng(to_vector(Plocal[sim_states[parent]]));
    }
    
    array[N] int<lower=0, upper=1> x_prior;
    array[N] int<lower=0, upper=1> y_prior;
    for (i in 1:N) {
        int state = sim_states[i] - 1;
        x_prior[i] = state / 2;
        y_prior[i] = state % 2;
    }

    
    // Posterior predictive for each tip
    array[N] real y_log_lik;  // Pointwise log-likelihood
    vector[4] ll_loo;
    array[N] int<lower=0, upper=1> x_rep;
    array[N] int<lower=0, upper=1> y_rep;
    for (tip in 1:N) {
        for (s in 1:4) {
            // Recompute likelihood WITHOUT this tip
            matrix[Nnodes, 4] loglik_loo = rep_matrix(0.0, Nnodes, 4);
            
            // Set tip likelihoods (excluding tip 'tip')
            for (i in 1:N) {
                loglik_loo[i] = rep_row_vector(negative_infinity(), 4);
                if (i != tip) {
                    int idx = 2 * x[i] + y[i] + 1;
                    loglik_loo[i, idx] = 0;
                } else {
                    loglik_loo[i, s] = 0;
                }
            }
            
            // Pruning recursion (same as model block)
            for (e in 1:Nedges) {
                int parent = edges[e, 1];
                int child = edges[e, 2];
                real t = edge_lengths[e];
                matrix[4, 4] Plocal = matrix_exp(Q * t);
                
                for (k in 1:4) {
                    loglik_loo[parent, k] += log_sum_exp(
                        to_vector(log(Plocal[k]) + loglik_loo[child])
                    );
                }
            }
            
            ll_loo[s] = log_sum_exp(loglik_loo[root_node] + log(stat_dist)');
        }
        int observed_state = 2 * x[tip] + y[tip] + 1;
        int same_x_state = 2 * x[tip] + (1 - y[tip]) + 1;
        y_log_lik[tip] = ll_loo[observed_state] - log_sum_exp(ll_loo[observed_state] , ll_loo[same_x_state]);
        int s_rep = categorical_rng(softmax(ll_loo));
        x_rep[tip] = (s_rep - 1) / 2;
        y_rep[tip] = (s_rep - 1) % 2;
    }
}
