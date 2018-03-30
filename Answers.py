import CalibrationClasses as CalibClasses
import CalibrationSettings as CalibSets
import scr.FigureSupport as Fig
import scipy.stats as stat


print ('Problem 2')
print ('Binomial distribution with p (probability of successes) equals the probability of surviving beyond 5 years (q) and number of trials as N.')
print ()
print ('Problem 3')
prob=stat.binom.pmf(400, 573, 0.5)
print('The likelihood that a clinical study reports 400 of 573 participants survived at the end of the 5-year study period '
      'if 50% of the patients in our simulated cohort survived beyond 5 years is:', prob)
print()
print("Problem 4")

# create a calibration object
calibration = CalibClasses.Calibration()

# sample the posterior of the mortality probability
calibration.sample_posterior()

# create the histogram of the resampled mortality probabilities
Fig.graph_histogram(
    observations=calibration.get_mortality_resamples(),
    title='Histogram of Resampled Mortality Probabilities',
    x_label='Mortality Probability',
    y_label='Counts',
    x_range=[CalibSets.POST_L, CalibSets.POST_U])

# Estimate of mortality probability and the posterior interval
print('Estimate of mortality probability ({:.{prec}%} credible interval):'.format(1-CalibSets.ALPHA, prec=0),
      calibration.get_mortality_estimate_credible_interval(CalibSets.ALPHA, 4))
print()
print('Problem 5')
# initialize a calibrated model
calibrated_model = CalibClasses.CalibratedModel('CalibrateResults.csv')
# simulate the calibrated model
calibrated_model.simulate(CalibSets.SIM_POP_SIZE, CalibSets.TIME_STEPS)

# plot the histogram of mean survival time
Fig.graph_histogram(
    observations=calibrated_model.get_all_mean_survival(),
    title='Histogram of Mean Survival Time',
    x_label='Mean Survival Time (Year)',
    y_label='Count',
    x_range=[10, 20])

# report mean and projection interval
print('Mean survival time and {:.{prec}%} projection interval:'.format(1 - CalibSets.ALPHA, prec=0),
      calibrated_model.get_mean_survival_time_proj_interval(CalibSets.ALPHA, deci=4))

print()
print('Problem 6')
print('credible interval of the estimated annual mortality probability and in the projection interval of the mean survival time both get narrower.')


