# saved as model_jags.R
model {
  for (i in 1:N) {
    ### likelihoods
    D[i] ~ dbin(IFR[i], C[i]) # number of deaths in cases C
    C[i] ~ dbin(IR[i], P[i]) # number of cases in population P
    CC[i] ~ dbin(p[i], T[i]) # number of positive cases in tests T
    ### deterministics
    logit(IFR[i]) = IFR_normal[i]
    logit(IR[i]) = IR_normal[i]
    logit(p[i]) = IR_normal[i] + log(bias)
    ### priors
    IFR_normal[i] ~ dnorm(IFR_mu, IFR_prec)
    IR_normal[i] ~ dnorm(IR_mu, IR_prec)
  }
  ### hyperpriors
  IFR_mu = logit(IFR_mu_raw)
  IFR_mu_raw ~ dunif(0, 1)
  IFR_prec ~ dnorm(0.0, 1.0 / 10.0) I(0, )
  IR_mu = logit(IR_mu_raw)
  IR_mu_raw ~ dunif(0, 1)
  IR_prec ~ dnorm(0.0, 1.0 / 10.0) I(0, )
  bias ~ dnorm(1.0, 1.0/0.4^2.0) I(1, )
}
