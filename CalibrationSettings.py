import scipy.stats as stat


SIM_POP_SIZE = 1000       # population size of simulated cohorts
TIME_STEPS = 1000        # length of simulation
ALPHA = 0.05             # significance level for calculating confidence intervals
NUM_SIM_COHORTS = 500   # number of simulated cohorts used to calculate prediction intervals

# details of a clinical study estimating the annual mortality
OBS_TOTAL = 573       # number of patients involved in the study
OBS_FIVEYEAR_SURV = 400    # observed number of patients survived after 5 years
OBS_ALPHA = 0.05   # significance level


# how to sample the posterior distribution of mortality probability
# minimum, maximum and the number of samples for the mortality probablity
POST_L, POST_U, POST_N = 0.05,0.25, 1000
