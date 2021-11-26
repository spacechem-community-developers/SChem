#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import math
import random
import sys
from typing import Union

from scipy.stats import binom
# Stupid black magic to suppress rare `divide by zero encountered in _binom_cdf` warning when calling binom for first
# time with very particular values; scipy version 1.7.1. Python suppresses warnings past the first so we purposely
# trigger it here. Messing with warnings management didn't help since reverting them to normal resets the count.
import os
STDERR = sys.stderr
sys.stderr = open(os.devnull, 'w')
binom.cdf(39, 43097, 0.5)  # No, it doesn't occur for 38 or 40, or for any n lower than 43097
sys.stderr = STDERR

from .components import RandomInput
from .schem_random import SChemRandom

NON_PRECOG_MIN_PASS_RATE = 0.2

# We will keep two confidence levels (CLs) for statistical operations; a more strict ideal CL, which we will attempt
# to achieve if given enough time, and a fallback minimum acceptable CL, which if time-constrained, we will consider
# acceptable for returning an answer anyway even if the ideal is not achieved. If neither of these confidence levels
# is obtained, an error will be raised

# The preferred rate of precognitive solutions being marked as non-precognitive
# Lowering this increases the number of runs non-precog solutions require
# We could probably be more lax on this one since precogs are submitted much less often, but it only saves about
# 10 runs when checking a typical non-precog production solution
PREFERRED_FALSE_NEG_RATE = 0.001
# The maximum acceptable rate of non-precognitive solutions being marked as precognitive
# Lowering this increases the number of runs precog solutions require
# This is the expensive one, but non-precogs are more common so we need a low false positive rate for them
PREFERRED_FALSE_POS_RATE = 0.001

# Fallback confidence levels used for very slow solutions if we can't run enough to reach the higher confidence levels
MAX_FALSE_POS_RATE = 0.1
MAX_FALSE_NEG_RATE = 0.1
# A constant factor that determines how quickly we decide a molecule variant has been assumed, if we see it fail X times
# without ever succeeding. We declare precog if the variant's success rate is provably (to within our above
# false positive confidence level) less than this ratio of the solution's observed success rate. Check the comments near
# its usage for a fuller explanation, but I don't believe it actually has to be near 0, and putting it near 0 scales the
# time precog solutions take to evaluate. For example, at a factor of 0.75 and false positive rate 0.001, for a solution
# that was originally observed to have 100% success rate before searching for missing variants, it will only be
# declared precog if a variant of the Nth molecule appears in 8 failing runs without ever appearing in a succeeding run.
# We do want it to be less than 1 however since that saves us from edge case handling if the success rate was originally
# measured to be 100%
MOLECULE_SUCCESS_RATE_DEVIATION_LIMIT = 0.75

# Since long cycle counts go hand-in-hand with demanding many runs for sufficient certainty, practical applications
# don't have time to properly check precog for long solutions. By default, cut off the max total cycles runtime and
# raise an error if this will be exceeded (rather than returning an insufficiently-confident answer)
DEFAULT_MAX_PRECOG_CHECK_CYCLES = 2_000_000  # Large enough to ensure it doesn't constrain typical required run counts


# TODO: Might want type hinting here, this post suggests a way to type hint Solution without introducing a circular
#       import or needing to merge the modules:
#       https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
def is_precognitive(solution, max_cycles=None, just_run_cycle_count=0, max_total_cycles=None,
                    include_explanation=False) -> Union[bool, tuple]:
    """Run this solution enough times to check if fits the community definition of a precognitive solution.

    If time constraints do not allow enough runs for even 90% certainty either way, raise a TimeoutError.

    Currently, a solution is considered precognitive if:
    * It assumes the value of the Nth molecule of a random input, for some N >= 2.
      Stated conversely, a solution (with acceptable success rate) is non-precognitive if, for each random input I,
      each N >= 2, and each type of molecule M that I produces, there exists a random seed where the Nth input of I is
      M, and the solution succeeds.
    * OR it succeeds for < 20% of random seeds.
      Accordingly with the first rule excepting the first input molecule, this check only uses seeds that match
      the first molecule (or all first molecules if there are multiple random inputs), if that is more favourable.

    In practice we check this with the following process:
    1. Run the solution on the original level, verifying it succeeds (validate the solution's expected score here too if
       possible). Track how many molecules were generated from each random input (call this M), and what the mth
       molecule's variant was for every m up to M.
    2. Randomize the input seed (in the case of two random inputs, shift the second seed by the same amount).
    3. Repeat step 1 with the new random seed(s) (but without requiring that the run succeed). Update M for each random
       input to be the minimum number of molecules produced from that input for any passing run (since any unconsumed
       input cannot have been assumed). Once again track the molecule variants that appeared, keeping a tally of how
       many times each variant has been in a passing vs failing run.
    4. Repeat steps 2-3 until any of the following conditions is met (again ignoring seeds that had a differing first
       molecule if that is more forgiving):
       * The success rate is measured to be < 20%, with 99.9% confidence (precog).
       * The success rate is measured to be > 20% with 99.9% confidence, and the dataset of succeeding runs covers
         every possible variant of every possible mth molecule (2 <= m <= M), for all random inputs (non-precog).
       * A variant of the mth molecule fails sufficiently many runs without ever succeeding (precog).
         This threshold is calculated dynamically based on the observed success rate.
       * The maximum allowed runs based on max_total_cycles is reached (TimeoutError).
         With default settings this should only occur for very long (100k+ cycles) solutions or solutions with a
         failure rate extremely close to 20%.

    Args:
        solution: The loaded solution to check.
        max_cycles: The maximum cycle count allowed for a SINGLE run of the solution (passed to Solution.run).
            Note that this is not the total number of cycles allowed across all runs; any solution within this limit
            is allowed to run at least twice, with the maximum runs taken being limited for extremely slow solutions.
        max_total_cycles: The maximum total cycle count that may be used by all runs; if this value is exceeded before
            sufficient confidence in an answer is obtained, a TimeoutError is raised.
        just_run_cycle_count: In order to save on excess runs, if the solution has just been successfully run on the
            loaded level (and not been modified or reset() since), pass its cycle count here to skip the first run (but
            still pull the first run's data from the Solution object).
        include_explanation: If True, instead of the boolean result, return a tuple of (result, explanation), where
            the latter is a string describing why the solution was or was not determined to be precognitive.
    """
    # Hang onto references to each random input in the solution
    random_inputs = [input_component for input_component in solution.inputs
                     if isinstance(input_component, RandomInput)]

    if not random_inputs:  # duh
        return (False, "Solution is not precognitive; level is non-random") if include_explanation else False

    # Set a larger default for max_cycles than in Solution.run, since the seed might change the cycle count by a lot
    if max_cycles is None and solution.expected_score is not None:
        max_cycles = 2 * solution.expected_score.cycles

    if max_total_cycles is None:
        # TODO: Might also want to limit this by reactor count
        max_total_cycles = DEFAULT_MAX_PRECOG_CHECK_CYCLES

    total_cycles = 0
    # Track the min cycles a passing run takes so we can exit early if we know we can't prove anything before timeout
    min_passing_run_cycles = math.inf

    # For each input zone, let M be the minimum molecules the solution must use from that input to succeed
    # Before we do any checks that require resetting the input objects, initialize Ms to the data from the last run if
    # just_run_cycle_count was provided
    # Ignore molecules that only made it into the pipe since they can't affect the solution
    # TODO: Find a way to share this code with the same in-loop calculation
    Ms = [random_input.num_inputs - len(random_input.out_pipe._molecules)
          if just_run_cycle_count else math.inf
          for random_input in random_inputs]

    # If the solution didn't use any of the random inputs, it never will
    if all(M == 0 for M in Ms):
        return (False, "Solution is not precognitive; does not use random inputs.") if include_explanation else False

    # Collect a bunch of information about each random input which we'll use for calculating how many runs are needed
    num_variants = [len(random_input.molecules) for random_input in random_inputs]
    first_input_variants = [random_input.reset().get_next_molecule_idx() for random_input in random_inputs]
    # When accounting for the allowable first input assumption, we need to know whether that variant will be impossible
    # to find for the rest of its bucket
    first_input_is_unique = [random_input.input_counts[first_variant] == 1
                                for random_input, first_variant in zip(random_inputs, first_input_variants)]
    bucket_sizes = [sum(random_input.input_counts) for random_input in random_inputs]

    # Global run counters. These include data from runs biased by seed-skipping, so are mostly for reporting purposes
    num_runs = 0
    num_passing_runs = 0  # TODO: Unused atm but I might need this if I add more detailed --debug prints
    # Runs where the first molecule of each input matched that in the base seed
    num_runs_first_match = 0
    num_passing_runs_first_match = 0

    # For performance reasons, once certain checks pass/fail, we start searching for seeds that are relevant to the
    # remaining checks (i.e. particular molecules appear). This biases run success rates.
    # We also consider the first run to be biased, since it's the seed a player engineers the solution to succeed for.
    # Keep some variables tracking the state of relevant checks, and extra counters that count runs of various bias
    # levels.
    global_success_check_failed = False  # If the global success rate fails, we start focusing on first-matching runs
    global_success_check_succeeded = False  # Only used for more precise post-run reports
    success_check_passed = False  # Once either success rate check passes, we start focusing on particular variants

    # Runs for which no seed skipping of any sort was done
    num_runs_unbiased = 0
    num_passing_runs_unbiased = 0
    # Runs where the first molecule of each input matched that in the base seed, either unbiased or with the
    # first molecules having been forced via seed skipping (but with no other forced molecules)
    num_runs_first_match_unbiased = 0
    num_passing_runs_first_match_unbiased = 0

    expl = ""  # Var to allow sub-functions to add to the result explanation or for it to be expanded on piecemeal

    def global_success_rate_okay(false_neg_rate):
        """Check if, with sufficient confidence, the all-seeds success rate is high enough."""
        # Using the binomial cumulative distribution function of the failure count (= P(failures <= X)), and assuming
        # the highest disallowed success rate, check if the probability of seeing this few failures is below our false
        # negative threshold.
        return binom.cdf(num_runs_unbiased - num_passing_runs_unbiased,
                         num_runs_unbiased,
                         1 - NON_PRECOG_MIN_PASS_RATE) < false_neg_rate

    def first_match_success_rate_okay(false_neg_rate):
        """Check if, with sufficient confidence, the success rate is high enough for first-molecule-matching seeds."""
        return binom.cdf(num_runs_first_match_unbiased - num_passing_runs_first_match_unbiased,
                         num_runs_first_match_unbiased,
                         1 - NON_PRECOG_MIN_PASS_RATE) < false_neg_rate

    def success_rate_okay(false_neg_rate):
        """Check if, with sufficient confidence, the success rate is high enough, for either all seeds or all seeds
        where the first molecule matched.
        """
        nonlocal global_success_check_succeeded
        if not global_success_check_succeeded:
            global_success_check_succeeded = global_success_rate_okay(false_neg_rate)

        return global_success_check_succeeded or first_match_success_rate_okay(false_neg_rate)

    def global_success_rate_too_low(false_pos_rate):
        """Check if, with sufficient confidence, the success rate is too low for all seeds."""
        # Using the binomial cumulative distribution function of the success count (= P(successes <= X)), and assuming
        # the lowest allowed success rate, check if the probability of seeing this few successes is below our false
        # positive threshold.
        return binom.cdf(num_passing_runs_unbiased, num_runs_unbiased, NON_PRECOG_MIN_PASS_RATE) < false_pos_rate

    def first_match_success_rate_too_low(false_pos_rate):
        """Check if, with sufficient confidence, the success rate is too low for seeds with the same first molecule
        as the base seed.
        """
        return (binom.cdf(num_passing_runs_first_match_unbiased, num_runs_first_match_unbiased,
                          NON_PRECOG_MIN_PASS_RATE)
                < false_pos_rate)

    def success_rate_too_low(false_pos_rate):
        """Check if, with sufficient confidence, the success rate is too low for both categories of seeds."""
        # If the global success check fails, update the state so we can start seed-skipping to finish the other check
        nonlocal global_success_check_failed
        if not global_success_check_failed:
            global_success_check_failed = global_success_rate_too_low(false_pos_rate)

        # Since assuming the first input is allowed, the rate is only considered too low if it's too low both including
        # and not including seeds that match the base seed's first molecule(s)
        if global_success_check_failed and first_match_success_rate_too_low(false_pos_rate):
            if include_explanation:
                nonlocal expl
                # Since it's a pretty common case, we'll simplify the message if everything failed
                if num_passing_runs_unbiased == num_passing_runs_first_match_unbiased == 0:
                    expl += (f"Solution is precognitive; <= {round(100 * NON_PRECOG_MIN_PASS_RATE)}% success rate for a"
                             f" random seed (with {100 * (1 - false_pos_rate)}% confidence); all {num_runs - 1}"
                             f" alternate-seed runs failed.")
                else:
                    success_rate = num_passing_runs_unbiased / num_runs_unbiased
                    success_rate_first_match = num_passing_runs_first_match_unbiased / num_runs_first_match_unbiased
                    expl += (f"Solution is precognitive; <= {round(100 * NON_PRECOG_MIN_PASS_RATE)}% success rate for a"
                             f" random seed (with {100 * (1 - false_pos_rate)}% confidence);"
                             f" {round(100 * success_rate)}% of {num_runs_unbiased} alternate-seed runs passed"
                             f" (or {round(100 * success_rate_first_match)}% of {num_runs_first_match_unbiased} runs"
                             " when targeting seeds with same first molecule as the base seed).")

            return True

        return False

    # For each random input, track which variants of the Nth molecule have been seen, and how many runs it passed vs
    # failed. We could get away with Sets instead of Counters for the success data but I prefer to keep things symmetric
    # Since we allow first input assumptions, we don't store the 1st input's variants, but store a dummy
    # value at the front to keep our indices sane
    # Success data could get away with just set() but the symmetry keeps the code cleaner for negligible extra memory
    success_run_variants = [[Counter()] for _ in range(len(random_inputs))]
    # TODO: This var is unused but keeps the code symmetrical; rip it out without hurting the symmetry too much.
    #       Probably switching success data back to set() at the same time will make the two asymmetries mostly cancel out
    fail_run_variants = [[Counter()] for _ in range(len(random_inputs))]

    # Keep additional datasets that track only data from runs that had the same first molecule(s) as the base seed, so
    # our checks are unbiased for solutions that use the allowable assumption on the first input
    # Note that we don't need a separate measure of Ms since it is a minimum of any successful run, regardless of seed
    success_run_variants_first_match = [[Counter()] for _ in range(len(random_inputs))]
    fail_run_variants_first_match = [[Counter()] for _ in range(len(random_inputs))]

    def check_molecule_assumptions(false_pos_rate, skip_non_precog_checks=False):
        """Return True if we can safely declare the solution assumes a particular molecule (other than the first),
        return False if we can safely declare it does not, and return None if we aren't confident either way yet.

        Also accept a flag to skip non-precog checks (saving a little computation) in the case that the success rate
        check hasn't passed yet. Checks that would determine the solution to be precog are still performed.
        """
        # TODO: skip_non_precog_checks flag is ugly, split this into two functions now that it's two independent blocks
        nonlocal expl

        # If for every random input, we've succeeded at least once on all molecule variants (ignoring the first
        # molecule) up to the minimum number of molecules the solution needs from that input to complete, there are
        # guaranteed no assumed molecules.
        if (not skip_non_precog_checks
            and all(len(success_run_variants[i][m]) == num_variants[i]
                    # To account for the allowed first molecule assumption, ignore the first molecule's variant in its
                    # bucket if it was unique, since it can be impossible for it to show up again.
                    or (first_input_is_unique[i]
                        and m < bucket_sizes[i]
                        and len(success_run_variants[i][m]) == num_variants[i] - 1
                        and first_input_variants[i] not in success_run_variants[i][m])
                    for i, M in enumerate(Ms)
                    for m in range(1, M))):  # Ignore first molecule
            if include_explanation:
                # We won't try to explain all the data bias-handling going on to the user; just report whichever
                # unbiased success rate passed the check, as well as the actual number of runs used in case they're
                # wondering why it's slow
                if global_success_check_succeeded:
                    success_rate = num_passing_runs_unbiased / num_runs_unbiased
                    expl += ("Solution is not precognitive; successful variants found for all input molecules in"
                             f" {num_runs} runs ({round(100 * success_rate)}% success rate).")
                else:
                    success_rate_first_match = num_passing_runs_first_match_unbiased / num_runs_first_match_unbiased
                    expl += ("Solution is not precognitive; successful variants found for all input molecules in"
                             f" {num_runs} runs ({round(100 * success_rate_first_match)}% success rate for seeds with"
                             f" same first molecule as base seed).")

            return False

        # Otherwise, check if any of the variants has failed X times without ever succeeding.
        # Before calculating X, we need to know what confidence level to use for its calculation.
        # To account for the increased chance of a false positive caused by individually testing every variant (e.g. for
        # a 50/50 production level, we'd have 80 individual variants to check, hence 80 chances for a variant to only
        # show up during failing runs by pure bad luck), we do a little rejiggering with some basic math:
        # total_false_positive_rate = P(any variant false positives)
        # = 1 - P(no variant false positives)
        # TODO: This is too strict because failures between variants aren't independent, since a single failing run
        #       increases the count of e.g. 40 variants - but they aren't fully correlated either... there's some
        #       deeper math to be done to reduce this exponent but for now we'll play it safe.
        # = 1 - P(single variant doesnt false positive)^total_variants
        # = 1 - (1 - P(single variant false positives))^total_variants
        # Rearranging, we get the stricter confidence level we must use for each individual variant check:
        # P(single variant false positives) = 1 - (1 - total_false_positive_rate)^(1 / total_variants)
        # Note that we don't care about the relative probabilities of the variants, since the solution has no control
        # over which variants it is tested on; more common molecule variants will be seen in successful runs sooner, but
        # the total probability that a variant eventually hits X failures before 1 success is the same as that for a
        # rarer variant, all things being equal.
        total_variants = sum(Ms[i] * num_variants[i] for i in range(len(random_inputs)))
        individual_false_pos_rate = 1 - (1 - false_pos_rate)**(1 / total_variants)
        # Now, to calculate X, consider that the solution has some unknown probability of succeeding for each given
        # variant of the Nth molecule (e.g. P(success | 3rd molecule is Nitrogen)). In order to declare the solution
        # precognitive, we must find one of these variants for which we can prove, with sufficient confidence, that
        # its probability of success equals 0.
        # However since proving an event is impossible is a hard problem (?), we'll settle for proving that a particular
        # molecule variant's success rate is statistically significantly far below some constant factor of the
        # solution's current success rate.
        # E.g. if the solution succeeds 90% of the time, we will be much more suspicious of always-failing variants than
        # if it succeeds 50% of the time. This isn't perfect since assumptions on sequences (allowed) might cause
        # certain molecules' variants to have a lower success rate, but my expectation is that this bias will be
        # somewhat counteracted by the stricter individual check confidence level (see above), since for
        # any single variant to be significantly below the average success rate, other variants must be above it, and
        # will thus have a reduced chance to false positive their own checks; making the effective confidence level
        # stricter in the worst case of 1-2 'biased but not assumed' variants.
        # In any case, this means we want (for some constant factor c < 1 that we'll pick to our liking):
        # P(false positive) <= P(X failures in X tries) = (1 - c * success_rate)^X
        # => X = log(P(false positive), base=(1 - c * success_rate))
        # Note that this becomes prohibitively large for success rates close to 0, but we restrict success rate anyway
        # so this is not a problem.
        # TODO: Might need to do two sets of checks, both with and without the off-seed runs

        # If we have yet to see a success (ignoring the biased first run), we cannot declare any variant to be failing
        # at a higher rate than normal, so skip this check (also avoids any log-base-1 errors)
        if num_passing_runs_first_match_unbiased == 0:
            return None

        # Note that we use the unbiased data from before we start seed-targeting variants still failing this
        # check, since otherwise we'd be aiming for a perpetually-moving target
        success_rate_first_match = num_passing_runs_first_match_unbiased / num_runs_first_match_unbiased
        max_variant_failures = math.ceil(math.log(individual_false_pos_rate,
                                                  1 - (MOLECULE_SUCCESS_RATE_DEVIATION_LIMIT
                                                       * success_rate_first_match)))
        # TODO: This check is doing much more work than needed since only newly-failing variants need to be re-checked
        #       It's insignificant compared to the cost of schem.run, but still.
        for i, M in enumerate(Ms):
            for m in range(1, M):
                if m >= len(fail_run_variants_first_match[i]):
                    break

                # TODO: Is there any point also analyzing failures caused by differing first molecule runs?
                #       No, but we CAN account for successes from off-seed runs. Of course, if a success appears in an
                #       off-seed run then we clearly haven't assumed the first input...
                for v in range(num_variants[i]):
                    if ((m > len(success_run_variants[i])
                         or v not in success_run_variants[i][m])
                            and fail_run_variants_first_match[i][m][v] >= max_variant_failures):
                        if include_explanation:
                            # Use the human-readable name for the variant if it's present and unique
                            # (for some levels, all input molecules have the same name which isn't very helpful)
                            mol_name = None
                            if len(set(mol.name for mol in random_inputs[i].molecules)) == num_variants[i]:
                                mol_name = random_inputs[i].molecules[v].name

                            if not mol_name:
                                mol_name = f"variant {v + 1}"  # 1-indexed for human-readability

                            expl += (f"Solution is precognitive; failed whenever molecule {m + 1} was {mol_name}, for"
                                     f" {max_variant_failures} such appearances (whereas solution success rate was"
                                     f" otherwise {round(100 * success_rate_first_match)}%).")

                        return True

        return None

    # Use a local random generator with fixed seed, to ensure results are reproducible
    rng = random.Random(0)
    first_seed = random_inputs[0].seed
    used_seeds = set()  # Track which seeds we've used (for the first input) to avoid duplicates
    # If there are multiple random inputs, keep the difference between their seeds fixed; in particular, we care about
    # ensuring that any random inputs that had the same seed will always be given the same seed, as there are currently
    # no rules against exploiting ramifications of this.
    input_seed_increments = [random_input.seed - first_seed for random_input in random_inputs]

    while total_cycles < max_total_cycles:
        # Randomize the seed
        # We've found that the SC RNG does not seem to be random enough when simply incrementing seeds;
        # instead choose the first input's seed randomly from all allowed seeds, using a static python RNG
        while first_seed in used_seeds:  # Note that this does nothing for the first run, as desired
            first_seed = rng.randint(0, SChemRandom.MAX_SEED)
        used_seeds.add(first_seed)

        # Set the other random input seeds to have the same increments off the first input's seed as originally
        for random_input, seed_increment in zip(random_inputs, input_seed_increments):
            random_input.seed = (first_seed + seed_increment) % (SChemRandom.MAX_SEED + 1)

        # Check if the first molecule of each random input is the same as in the original input
        first_molecule_matches = all(random_input.reset().get_next_molecule_idx() == first_input_variants[i]
                                     for i, random_input in enumerate(random_inputs))

        # If we're still working on the success rate check, but the any-seed portion of the check has already
        # definitively failed, skip this seed if the first molecule (of each input) doesn't match that of the base seed.
        if global_success_check_failed and not success_check_passed and not first_molecule_matches:
            continue

        # If we have already achieved sufficient confidence on the success rate, we are just waiting for all input
        # variants to show up and no longer need to worry about biasing the success rate; skip seeds to speed this
        # process up. Specifically, we will pick the variant we've seen fail the most runs without ever succeeding, and
        # skip all seeds not containing it. This will obviously speed up analysis of precog solutions, since the
        # assumed molecule's failing variant will quickly be specifically targeted to reach our desired failure count
        # threshold.
        # Additionally, compared to just skipping seeds where all variants have succeeded, this approach is also subtly
        # faster for non-precog solutions, because while the former will be indiscriminate in the number of failing
        # variants in the seed it picks (as long as there is at least one), the first seed to contain a particular
        # failing variant is more likely to be one that contains many failing variants, and thus it will tend to take
        # fewer runs to succeed every variant (this is true even if the chosen variant has an inherently higher failure
        # rate, since either way the expected number of runs containing that variant will be the same - there's just a
        # higher chance that when it succeeds, it also fulfills other variants).
        # Note that we can't directly require seeds with at least 2+ failing variants, because we could get locked into
        # continuously picking a particular sequence the solution (legally) assumes does not occur.
        # TODO: It'd be nice to not have to wait for the success rate check to be done to start skipping seeds...
        #       Possibly some skipping can be done that just fixes 'bad luck', and in theory only reduces
        #       volatility in the success rate measurement rather than really biasing it - for example, skipping
        #       an all-successful-variants seed if there exist variants that should have appeared by now but
        #       haven't due to bad luck.
        #       However this has to be implemented in a way that we're not forcing low-probability variants to appear
        #       at a rate higher than they should (or biasing their neighbor molecules' variants, etc...).
        # TODO 2: If the solution has never succeeded with a differing first molecule, it might be worth skipping those
        #         seeds too. However would have to be careful of bad luck causing us to never give it a chance to pass
        #         the off-brand seeds again.
        if success_check_passed:
            # Identify the input zone, molecule, and variant for which we've seen the most failures without a success
            target_input, target_molecule, target_variant = None, None, None
            max_variant_fail_count = -1
            for i, input_data in enumerate(fail_run_variants_first_match):
                for m in range(1, min(len(input_data), Ms[i])):
                    for v in range(num_variants[i]):
                        if ((m >= len(success_run_variants[i]) or v not in success_run_variants[i][m])
                                and input_data[m][v] > max_variant_fail_count):
                            target_input, target_molecule, target_variant = i, m, v
                            max_variant_fail_count = input_data[m][v]

            # If no runs have failed yet, pick the next variant we haven't seen succeed (ignoring first molecule)
            if target_input is None:
                # Awkward wrapped iterable to avoid having to break from a nested loop
                for i, m, v in ((i, m, v) for i in range(len(random_inputs))
                                          for m in range(1, Ms[i])
                                          for v in range(num_variants[i])):
                    if ((m >= len(success_run_variants[i]) or v not in success_run_variants[i][m])
                            # Make sure we don't pick a variant that's impossible under the first molecule assumption
                            and not (first_input_is_unique[i]
                                     and m < bucket_sizes[i]
                                     and v == first_input_variants[i])):
                        target_input, target_molecule, target_variant = i, m, v
                        break

            # Reset and skip past the molecules we don't care about
            random_inputs[target_input].reset()
            for n in range(target_molecule):
                random_inputs[target_input].get_next_molecule_idx()

            # Skip this seed if the target variant is not present
            if random_inputs[target_input].get_next_molecule_idx() != target_variant:
                continue

        solution.reset()  # Reset the solution from any prior run (this also picks up seed changes)

        # Run the solution with this seed of the input, checking if it succeeds (ignoring the exact score)
        try:
            # Run the solution
            # if just_run_cycle_count was given, skip the first run to save time
            cycles = just_run_cycle_count if num_runs == 0 and just_run_cycle_count \
                     else solution.run(max_cycles=max_cycles).cycles

            min_passing_run_cycles = min(min_passing_run_cycles, cycles)

            # Check how many molecules the solution consumed for each random input, and lower each M if possible
            # Note that if just_run_cycle_count was provided, we initialized M already and reset the solution,
            # so we skip in that case
            if not (num_runs == 0 and just_run_cycle_count):
                for i, random_input in enumerate(random_inputs):
                    # Ignore molecules that only made it into the pipe since their variant can't affect the solution
                    # TODO: Bad Pipe internal attribute access since pipes are not cycle-independent now, ditto above/below
                    this_M = random_input.num_inputs - len(random_input.out_pipe._molecules)
                    Ms[i] = min(Ms[i], this_M)

                # If the solution didn't use any of the random inputs, it never will (and if it did it always will)
                if num_runs == 0 and all(M == 0 for M in Ms):
                    if include_explanation:
                        return False, "Solution is not precognitive; does not use random inputs."
                    return False

            target_variants_data = success_run_variants
            target_variants_data_first_match = success_run_variants_first_match
            num_variants_to_store = Ms  # Direct reference is safe since we only read from this

            # Update relevant passing run counters
            num_passing_runs += 1
            # If we're still working on the global success check, we're not skipping seeds so the data is unbiased
            # (excepting the first run, which we always consider biased)
            if num_runs != 0 and not (global_success_check_failed or success_check_passed):
                num_passing_runs_unbiased += 1

            if first_molecule_matches:
                num_passing_runs_first_match += 1

                # If we're still working on either success check, we're not seed-skipping for variants other than the
                # first molecule, so first-match runs are unbiased (excepting the first run)
                if num_runs != 0 and not success_check_passed:
                    num_passing_runs_first_match_unbiased += 1
        except Exception as e:
            if num_runs == 0:
                # Not allowed to crash on the original seed, otherwise do nothing
                raise Exception(f"Error in base seed: {type(e).__name__}: {e}")

            cycles = solution.cycle
            target_variants_data = fail_run_variants  # The data set that this run's variants should be added to
            target_variants_data_first_match = fail_run_variants_first_match
            # Make sure we don't store data on variants of molecules from after the solution crashed
            num_variants_to_store = [random_input.num_inputs - len(random_input.out_pipe._molecules)
                                     for random_input in random_inputs]

        # Update run/cycle counters
        total_cycles += cycles
        num_runs += 1
        # If we're still working on the global success check, we're not skipping seeds so the data is unbiased
        # (excepting the first run, which we always consider biased)
        if num_runs != 1 and not (global_success_check_failed or success_check_passed):
            num_runs_unbiased += 1

        if first_molecule_matches:
            num_runs_first_match += 1

            # If we're still working on either success check, we're not seed-skipping for variants other than the
            # first molecule, so first-match runs are unbiased (excepting the first run)
            if num_runs != 1 and not success_check_passed:
                num_runs_first_match_unbiased += 1

        # Track all nth input variants that appeared in this run for 2 <= n <= N
        datasets_to_update = ([target_variants_data, target_variants_data_first_match]
                              if first_molecule_matches else
                              [target_variants_data])
        for dataset in datasets_to_update:
            for random_input, variants_data, num_new_variants in zip(random_inputs, dataset, num_variants_to_store):
                random_input.reset().get_next_molecule_idx()  # Reset and skip past n = 0

                if num_new_variants > len(variants_data):
                    variants_data.extend([Counter() for _ in range(num_new_variants - len(variants_data))])

                for n in range(1, num_new_variants):
                    variants_data[n][random_input.get_next_molecule_idx()] += 1

        # To save on futile runs, check if our time constraints can allow for us to get a sufficiently confident answer
        # about the success rate, assuming all future runs are successes or all are failures.
        # If we have so few runs that we can guarantee even our fallback confidence level won't be met, we can
        # immediately timeout
        remaining_cycles = max(max_total_cycles - total_cycles, 0)
        # The number of passing runs we're allowed before timing out is easy to estimate
        max_remaining_passing_runs = remaining_cycles // min_passing_run_cycles
        # On the other hand, the number of failing runs we're allowed before timing out is hard to estimate, since in
        # the case of a solution that assumes the first molecule, the seeds with non-matching first molecule will tend
        # to fail within just a few cycles, while matching seeds will usually fail at more random times.
        # However since the success rate check fails only if both the all-seeds and the first-matching seeds success
        # rates fail, it is sufficient to exit early only when it looks like the first-matching seeds success rate can't
        # fail, and these will tend not to fail immediately in either solution type (or if they do, the regular checks
        # will converge quickly enough that the time constraint won't be approached and this check won't matter).
        # Accordingly, we'll assume the failure runs average half a passing run's cycle count
        max_remaining_failing_runs = (2 * remaining_cycles) // min_passing_run_cycles
        if not (success_check_passed
                # Check if pure failures can confirm a too-low success rate (for first-match seeds)
                or binom.cdf(num_passing_runs_first_match_unbiased,
                             num_runs_first_match_unbiased + max_remaining_failing_runs,
                             NON_PRECOG_MIN_PASS_RATE)
                   < MAX_FALSE_POS_RATE
                # Check if pure successes can confirm a sufficiently high success rate (for either seed type)
                or binom.cdf(num_runs_unbiased - num_passing_runs_unbiased,
                             num_runs_unbiased + max_remaining_passing_runs,
                             1 - NON_PRECOG_MIN_PASS_RATE)
                   < MAX_FALSE_NEG_RATE
                or binom.cdf(num_runs_first_match_unbiased - num_passing_runs_first_match_unbiased,
                             num_runs_first_match_unbiased + max_remaining_passing_runs,
                             1 - NON_PRECOG_MIN_PASS_RATE)
                   < MAX_FALSE_NEG_RATE):
            raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                               f" too few runs to ascertain {round(100 * NON_PRECOG_MIN_PASS_RATE)}% success rate"
                               f" requirement ({num_passing_runs_unbiased} / {num_runs_unbiased} alternate-seed runs"
                               f" passed, or {num_passing_runs_first_match_unbiased} / {num_runs_first_match_unbiased}"
                               " when targeting seeds with same first molecule as the base seed).")

        if not success_check_passed:
            # Note that this helper updates global_success_check_failed if/when relevant
            if success_rate_too_low(false_pos_rate=PREFERRED_FALSE_POS_RATE):
                return (True, expl) if include_explanation else True
            elif success_rate_okay(false_neg_rate=PREFERRED_FALSE_NEG_RATE):
                success_check_passed = True

        mol_assumption_check_result = check_molecule_assumptions(
            # Skip non-precog exit conditions if we aren't confident in the success rate yet
            skip_non_precog_checks=not success_check_passed,
            false_pos_rate=PREFERRED_FALSE_POS_RATE)

        if mol_assumption_check_result is not None:
            return (mol_assumption_check_result, expl) if include_explanation else mol_assumption_check_result

    # If we escaped the run loop without returning, we've been time-constrained in our number of runs.
    # Attempt to redo the precog check with our fallback relaxed confidence levels, and if even then we aren't
    # sufficiently confident in our answer, raise an error
    if include_explanation:
        expl += "Warning: Precog check terminated early due to time constraints; check accuracy may be reduced.\n"

    if not success_check_passed:
        if success_rate_too_low(false_pos_rate=MAX_FALSE_POS_RATE):
            return (True, expl) if include_explanation else True

        if not success_rate_okay(false_neg_rate=MAX_FALSE_NEG_RATE):
            raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                               f" too few runs to ascertain {round(100 * NON_PRECOG_MIN_PASS_RATE)}% success rate"
                               f" requirement ({num_passing_runs_unbiased} / {num_runs_unbiased} alternate-seed runs"
                               f" passed, or {num_passing_runs_first_match_unbiased} / {num_runs_first_match_unbiased}"
                               " when targeting seeds with same first molecule as the base seed).")

    mol_assumption_check_result = check_molecule_assumptions(false_pos_rate=MAX_FALSE_POS_RATE)
    if mol_assumption_check_result is not None:
        return (mol_assumption_check_result, expl) if include_explanation else mol_assumption_check_result

    raise TimeoutError("Precog check could not be completed due to time constraints; certain non-succeeding molecule"
                       f" variants not encountered enough times in {num_runs} runs to be confident they always fail.")
