from enum import Enum
import scipy.stats as stat
import numpy as np
import scr.InOutFunctions as InOutSupport
import scr.StatisticalClasses as StatSupport
import scr.FormatFunctions as FormatSupport
import SurvivalModelClasses as SurvivalCls
import CalibrationSettings as CalibSets


class CalibrationColIndex(Enum):
    """ indices of columns in the calibration results cvs file  """
    ID = 0          # cohort ID
    W = 1  # likelihood weight
    MORT_PROB = 2   # mortality probability


class Calibration:
    def __init__(self):
        """ initializes the calibration object"""
        np.random.seed(1)   # specifying the seed of the numpy random number generator
        self._cohortIDs = range(CalibSets.POST_N)   # IDs of cohorts to simulate
        self._mortalitySamples = []      # values of mortality probability at which the posterior should be sampled
        self._mortalityResamples = []    # resampled values for constructing posterior estimate and interval
        self._weights = []               # likelihood weights of sampled mortality probabilities
        self._normalizedWeights = []     # normalized likelihood weights (sums to 1)
        self._csvRows = \
            [['Cohort ID', 'Likelihood Weights' ,'Mortality Prob']]  # list containing the calibration results

    def sample_posterior(self):
        """ sample the posterior distribution of the mortality probability """

        # find values of mortality probability at which the posterior should be evaluated
        self._mortalitySamples = np.random.uniform(low = CalibSets.POST_L,
                                                   high = CalibSets.POST_U,
                                                   size = CalibSets.POST_N)

        # create a multi cohort
        multiCohort = SurvivalCls.MultiCohort(ids = self._cohortIDs,
                                              mortality_probs = self._mortalitySamples,
                                              pop_sizes = [CalibSets.SIM_POP_SIZE]*CalibSets.POST_N)

        # simulate the multi cohort
        multiCohort.simulate(CalibSets.TIME_STEPS)

        # calculate the likelihood of each simulated cohort
        for cohort_id in self._cohortIDs:
            # get average survival time for this cohort:
            fiveYearSurv = multiCohort.get_five_year_survival(cohort_id)

            # construct a binormial distribution
            # fiveYearSurv = calculated from the simulated cohort
            weight = stat.binom.pmf(CalibSets.OBS_FIVEYEAR_SURV, CalibSets.OBS_TOTAL, fiveYearSurv)
            self._weights.append(weight)

        # normalize the likelihood weights
        sum_weights = np.sum(self._weights)
        self._normalizedWeights = np.divide(self._weights, sum_weights)

        # re-sample mortality probability (with replacement) according to likelihood weights
        self._mortalityResamples = np.random.choice (a = self._mortalitySamples,
                                                     size = CalibSets.NUM_SIM_COHORTS,
                                                     replace = True,
                                                     p = self._normalizedWeights)

        # produce the list to report the results
        for i in range (0, len(self._mortalitySamples)):
            self._csvRows.append([self._cohortIDs[i], self._normalizedWeights[i],
                                  self._mortalitySamples[i]])

        # write the calibration result into a csv file
        InOutSupport.write_csv('CalibrateResults.csv', self._csvRows)


    def get_mortality_resamples(self):
        """
        :return: mortality resamples
        """
        return self._mortalityResamples

    def get_mortality_estimate_credible_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of the posterior distribution"""

        # calculate the credible interval
        sum_stat = StatSupport.SummaryStat('Posterior samples', self._mortalityResamples)

        estimate = sum_stat.get_mean()  # estimated mortality probability
        credible_interval = sum_stat.get_PI(alpha)  # credible interval

        return FormatSupport.format_estimate_interval(estimate, credible_interval, deci)


class CalibratedModel:
    """ to run the calibrated survival model """

    def __init__(self, csv_file_name):
        """ extracts seeds, mortality probabilities and the associated likelihood from
        the csv file where the calibration results are stored
        :param cvs_file_name: name of the csv file where the calibrated results are stored
        """

        # read the columns of the csv files containing the calibration results
        cols = InOutSupport.read_csv_cols(file_name=csv_file_name, n_cols = 3,
                                          if_ignore_first_row=True,
                                          if_convert_float=True)

        # store likelihood weights, cohort IDs and sampled mortality probabilities
        self._cohortIDs = cols[CalibrationColIndex.ID.value].astype(int)
        self._weights = cols[CalibrationColIndex.W.value]
        self._mortality = cols[CalibrationColIndex.MORT_PROB.value]
        self._multiCohorts = None

    def simulate(self, cohort_size, time_steps):
        """ simulate the specified number of cohorts based on their associated likelihood weight
        :param cohort_size: the population size of cohorts
        """
        # resample cohort IDs and mortality probabilities based on their likelihood weights
        # sample (with replacement) from indices [0, 1, 2, ..., number of weights] based on the likelihood weights
        sample_row_indeces = np.random.choice(a = range(0, len(self._weights)),
                                              size = CalibSets.NUM_SIM_COHORTS,
                                              replace = True,
                                              p=self._weights)

        # use the sampled indices to populate the list of cohort IDs and mortality probabilities
        resample_ids=[]
        resampled_probs=[]
        for i in sample_row_indeces:
            resample_ids.append(self._cohortIDs[i])
            resampled_probs.append(self._mortality[i])

        # simulate the desired number of cohorts
        self._multiCohorts=SurvivalCls.MultiCohort(
            ids=resample_ids,
            pop_sizes=[cohort_size]*CalibSets.NUM_SIM_COHORTS,
            mortality_probs=resampled_probs
        )

        # simulate all cohorts
        self._multiCohorts.simulate(time_steps)


    def get_all_mean_survival(self):
        """ :returns a list of mean survival time for all simulated cohorts"""
        return self._multiCohorts.get_all_mean_survival()

    def get_mean_survival_time_proj_interval(self, alpha, deci):
        """
        :param alpha: the significance level
        :param deci: decimal places
        :returns text in the form of 'mean (lower, upper)' of projection interval
        """

        mean = self._multiCohorts.get_overall_mean_survival()
        proj_interval = self._multiCohorts.get_PI_mean_survival(alpha)

        return FormatSupport.format_estimate_interval(mean, proj_interval, deci)

