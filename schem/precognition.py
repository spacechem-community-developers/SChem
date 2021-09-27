#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from sys import stdout as STDOUT, stderr as STDERR

from scipy.stats import binom

from .solution import Solution
from .components import RandomInput

NON_PRECOG_MIN_PASS_RATE = 0.5

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

# Since long cycle counts go hand-in-hand with demanding many runs for sufficient certainty, practical applications
# don't have time to properly check precog for long solutions. By default, cut off the max total cycles runtime and
# raise an error if this will be exceeded (rather than returning an insufficiently-confident answer)
DEFAULT_MAX_PRECOG_CHECK_CYCLES = 2_000_000  # Large enough to ensure it doesn't constrain typical required run counts


def binary_int_search(f, low=0, high=1):
    """Given a boolean function f(n) of the form g(n) >= c, where g is monotonically increasing, use binary search to
    find the minimum natural value of n that solves the equation.
    """
    # Init: double high until we find an upper bound
    while not f(high):
        high *= 2

    while high != low + 1:
        mid = (high + low) // 2
        if f(mid):
            high = mid
        else:
            low = mid

    return high


def is_precognitive(solution: Solution, max_cycles=None, just_run_cycle_count=0, max_total_cycles=None,
                    verbose=False, stderr_on_precog=False):
    """Given a Solution, run/validate it then check if fits the current community definition of a precognitive solution.

    If time constraints do not allow enough runs for even 90% certainty either way, raise a TimeoutError.

    Currently, a solution is considered precognitive if:
    * It assumes the value of the Nth molecule of a random input, for some N >= 2.
      Stated conversely, a solution (with acceptable success rate) is non-precognitive if, for each random input I,
      each N >= 2, and each type of molecule M that I may create, there exists a random seed where the Nth input of I is
      M, and the solution succeeds.
    * OR it fails for >= 50% of random seeds.
         Accordingly with the first rule excepting the first input molecule, this check only uses seeds that match
         the first molecule (or all first molecules if there are multiple random inputs), if that is more favourable.

    In practice we check this with the following process:
    1. Run the solution on the original level, verifying it succeeds (validate the solution's expected score here too if
       possible). Track how many molecules were generated from each random input (call this N), and what the nth
       molecule's variant was for every n up to N.
    2. Increment the seeds of all random inputs at once until a set of seeds is found that has the same first molecule
       produced for each input (since assumptions on the first input are allowed). Incrementing the seeds in lockstep
       also ensures assumptions about inputs that share a seed are not violated (such assumptions are not forbidden in
       non-precognitive solutions).
    3. Repeat step 1 with the new random seed(s) (but without requiring that it succeed). Update N for each random input
       to be whichever run used less molecules (since the solution might always terminate on a certain molecule, using
       the max would risk the last n never being able to 'find' all variants).
       If the solution succeeds, aggregrate the nth-input-variant data with that of the first run, in order to track
       which variants for a given n have had at least one run succeed.
       If the solution fails, put the same nth-input-variant data in a separate 'failure variants' dataset.
    4. Repeat steps 2-3 until any of the following conditions is met (again ignoring seeds that had a differing first
       input if that is more forgiving):
       * The failure rate is measured to be >= 50%, with sufficient confidence (precog).
       * The success rate is measured to be > 50% with sufficient confidence, and the dataset of succeeding runs covers
         every possible variant of every possible n (2 <= n <= N), for all random input components (non-precog).
       * The maximum allowed runs based on max_total_cycles is reached (TimeoutError).
         With default settings this should only occur for very long (100k+ cycles) solutions or solutions with a
         failure rate extremely close to 50%.

    Args:
        solution: The loaded solution to check.
        max_cycles: The maximum cycle count allowed for a SINGLE run of the solution (passed to Solution.run).
            Note that this is not the total number of cycles allowed across all runs; any solution within this limit
            is allowed to run at least twice, with the maximum runs taken being limited for extremely slow solutions.
            This is bounded to 300 runs, since at a certain point it can be assumed that the missing variants are not
            being found because the first molecule was unique in its bucket.
        max_total_cycles: The maximum total cycle count that may be used by all runs; if this value is exceeded before
            sufficient confidence in an answer is obtained, a TimeoutError is raised.
        just_run_cycle_count: In order to save on excess runs, if the solution has just been successfully run on the
            loaded level (and not been modified or reset() since), pass its cycle count here to skip the first run (but
            still pull the first run's data from the Solution object).
        verbose: If True, print more detailed info on the number of runs performed and how many passed before returning.
        stderr_on_precog: If True, when a solution is precognitive, print an explanation of why to STDERR.
                          Can be enabled independently of `verbose`. Default False.
    """
    if max_cycles is None:
        if solution.expected_score is not None:
            # Larger default than in Solution.run since the seed might change the cycle count by a lot
            max_cycles = 2 * solution.expected_score.cycles
        else:
            max_cycles = Solution.DEFAULT_MAX_CYCLES

    if max_total_cycles is None:
        # TODO: Might also want to limit this by reactor count
        max_total_cycles = DEFAULT_MAX_PRECOG_CHECK_CYCLES

    total_cycles = 0
    # Track the min cycles a passing run takes so we can exit early if we know we can't prove anything before timeout
    min_passing_run_cycles = math.inf

    # Hang onto references to each random input in the solution
    random_inputs = [input_component for input_component in solution.inputs
                     if isinstance(input_component, RandomInput)]

    if not random_inputs:
        return False  # duh

    # For each input, N is the minimum molecules the solution must use from that input
    # Before we do any checks that require resetting the input objects, initialize Ns to the data from the last run if
    # just_run_cycle_count was provided
    Ns = [random_input.num_inputs if just_run_cycle_count else math.inf for random_input in random_inputs]

    # Collect a bunch of information about each random input which we'll use for calculating how many runs are needed
    num_variants = [len(random_input.molecules) for random_input in random_inputs]
    first_input_variants = [random_input.reset().get_next_molecule_idx() for random_input in random_inputs]
    bucket_sizes = [sum(random_input.input_counts) for random_input in random_inputs]
    rare_counts = [min(random_input.input_counts) for random_input in random_inputs]  # Bucket's rarest molecule count
    # Rarest molecule count in first bucket, accounting for the fixed first molecule
    # and ignoring now-impossible variants (since they can't fail a run)
    first_bucket_rare_counts = [min(count - 1 if variant == first_variant else count
                                    for variant, count in enumerate(random_input.input_counts)
                                    if (count - 1 if variant == first_variant else count) != 0)
                                for random_input, first_variant in zip(random_inputs, first_input_variants)]

    # Calculate the min number of runs seemingly non-precog solutions must take, before we're confident an
    # illegally-assumed molecule would have had its failing variant show up by now (i.e. we haven't false-negatived
    # a precog solution). Note that this isn't an absolute minimum on the number of runs since finding all variants is
    # still a sufficient condition to declare a solution non-precognitive, but this allows us to exit early
    # for solutions for which we still haven't found all variants (if none of the missing variants failed).
    # This is based on re-arranging (where n is the index of a molecule the solution illegally assumes):
    # P(nth input has any untested variant)
    # <= (1 - rarest_variant_chance)^runs
    # <= PREFERRED_FALSE_NEG_RATE
    min_early_exit_runs = 1
    fallback_min_early_exit_runs = 1  # Used if we get time constrained
    for i, random_input in enumerate(random_inputs):
        # Since we only check seeds with same first molecule as the base seed to respect the first input assumption,
        # molecules from the first bucket may end up with a rarer variant. Use the worst case (ignoring cases that
        # become impossible in the first bucket since those can also never cause a run to fail)
        rarest_variant_chance = min(rare_counts[i] / bucket_sizes[i],
                                    first_bucket_rare_counts[i] / (bucket_sizes[i] - 1))

        min_early_exit_runs = max(min_early_exit_runs, math.ceil(math.log(PREFERRED_FALSE_NEG_RATE)
                                                                 / math.log(1 - rarest_variant_chance)))
        fallback_min_early_exit_runs = max(fallback_min_early_exit_runs, math.ceil(math.log(MAX_FALSE_NEG_RATE)
                                                                                   / math.log(1 - rarest_variant_chance)))

    # For each random input, keep a list of sets containing all variants that appeared for the i-th input of that
    # random input. We don't store the 1st input's variants so store a dummy value at the front to keep our indices sane
    success_run_variants = [[set()] for _ in range(len(random_inputs))]
    fail_run_variants = [[set()] for _ in range(len(random_inputs))]
    num_runs = 0
    num_passing_runs = 0

    # Keep additional datasets that track only data from runs that had the same first molecule(s) as the base seed, so
    # our checks are unbiased for solutions that use the allowable assumption on the first input
    # Note that we don't need a separate measure of Ns since it is a minimum of any successful run, regardless of seed
    success_run_variants_first_match = [[set()] for _ in range(len(random_inputs))]
    fail_run_variants_first_match = [[set()] for _ in range(len(random_inputs))]
    num_runs_first_match = 0
    num_passing_runs_first_match = 0

    # Since in the case of a timeout we'll need to redo the precog checks with a different confidence level,
    # define some helpers for the precog checks

    def success_rate_okay(false_neg_rate):
        """Check if, with sufficient confidence, the success rate is high enough."""
        # Using the binomial cumulative distribution function of the failure count (= P(failures <= X)), and assuming
        # the highest disallowed success rate, check if the probability of seeing this few failures is below our false
        # negative threshold.
        # First run is ignored since it's biased (and must always pass).
        # Since assuming the first input is allowed, the check can pass either including or not including seeds that
        # match the base seed's first molecule(s)
        return (binom.cdf(num_runs_first_match - num_passing_runs_first_match, num_runs_first_match - 1,
                          1 - NON_PRECOG_MIN_PASS_RATE) < false_neg_rate
                or binom.cdf(num_runs - num_passing_runs, num_runs - 1, 1 - NON_PRECOG_MIN_PASS_RATE) < false_neg_rate)

    def success_rate_too_low(false_pos_rate):
        """Check if, with sufficient confidence, the success rate is too low."""
        # Using the binomial cumulative distribution function of the success count (= P(successes <= X)), and assuming
        # the lowest allowed success rate, check if the probability of seeing this few successes is below our false
        # positive threshold.
        # First run is ignored since it's biased (and must always pass).
        # Since assuming the first input is allowed, the rate is only considered too low if it's too low both including
        # and not including seeds that match the base seed's first molecule(s)
        if (binom.cdf(num_passing_runs_first_match - 1, num_runs_first_match - 1, NON_PRECOG_MIN_PASS_RATE)
                < false_pos_rate
            and binom.cdf(num_passing_runs - 1, num_runs - 1, NON_PRECOG_MIN_PASS_RATE)
                < false_pos_rate):
            if verbose or stderr_on_precog:
                print(f"Solution is precognitive; <= {100 * NON_PRECOG_MIN_PASS_RATE}% success rate for a random seed"
                      f" (with {100 * (1 - false_pos_rate)}% confidence);"
                      f" {num_passing_runs} / {num_runs} runs passed"
                      f" (or {num_passing_runs_first_match} / {num_runs_first_match} for seeds with same first molecule).",
                      file=STDERR if stderr_on_precog else STDOUT)

            return True

        return False

    max_success_runs = None  # Not calculated until we reach the minimum early exit condition

    def calc_max_success_runs(Ns, false_pos_rate):
        """Calculate the number of passing runs we must see before we can declare a molecule has been assumed."""
        # This is based on:
        # P(false positive)
        # ~= P(any molecule is missing a success variant)
        # = 1 - P(no molecule missing success variant)
        # = 1 - P(single molecule has all success variants)^(N - 1)
        # = 1 - P(first bucket molecule has all success variants)^(bucket_size - 1)
        #       * P(normal bucket molecule has all success variants)^(N - bucket_size)
        # ... (and so on) ...
        # ~= 1 - [1 - (1 - (first_bucket_rare_count / bucket_size))^successful_runs]^(bucket_size - 1)
        #        * [1 - (1 - (regular_bucket_rare_count / bucket_size))^successful_runs]^(N - bucket_size)
        # <= MAX_FALSE_POSITIVE_RATE
        # With the first bucket bias accounting, this is too complex to solve exactly for successful_runs
        # unlike with the false negative equation, but since we know it's monotonically increasing, we can just
        # binary search to find the minimum valid successful_runs
        max_success_runs = 1
        for i, (bucket_size, N) in enumerate(zip(bucket_sizes, Ns)):
            max_success_runs = max(max_success_runs, binary_int_search(
                lambda success_runs: 1 -
                                     # First bucket
                                     (1 - (1 - (first_bucket_rare_counts[i] / (bucket_size - 1)))
                                      ** success_runs)
                                     ** min(bucket_size - 1, N)  # N might be less than the bucket size
                                     # Remaining buckets
                                     * (1 - (1 - (rare_counts[i] / bucket_size))
                                        ** success_runs)
                                     ** max(N - bucket_size, 0)  # N might be less than the bucket size
                                     <= false_pos_rate))

        return max_success_runs

    def check_molecule_assumptions(min_early_exit_runs, false_pos_rate, skip_non_precog_checks=False):
        """Return True if we can safely declare the solution assumes a particular molecule (other than the first),
        return False if we can safely declare it does not, and return None if we aren't confident either way yet.

        Also accept a flag to skip non-precog checks in the case that the success rate check hasn't passed yet.
        """
        nonlocal max_success_runs  # We'll update this once after we've reached min_early_exit_runs

        # If for every random input, we've succeeded on all variants up to the minimum number of molecules the solution
        # needs from that input to complete, there are guaranteed no assumed molecules
        # It is sufficient to fulfill this condition in either the subset of seeds matching the base seed's first
        # molecules, or for the aggregrate data from all seeds
        if (not skip_non_precog_checks
            and (all(len(success_run_variants[i][n]) == num_variants[i]
                     for i, N in enumerate(Ns)
                     for n in range(1, N))
                 or all(len(success_run_variants_first_match[i][n]) == num_variants[i]
                        for i, N in enumerate(Ns)
                        for n in range(1, N)))):
            if verbose:
                print("Solution is not precognitive; successful variants found for all input molecules"
                      f" ({num_passing_runs} / {num_runs} runs passed, or"
                      f" {num_passing_runs_first_match} / {num_runs_first_match} for seeds with same first molecule).")

            return False

        # Since waiting until all variants show up is often overkill, we'll provide early exit conditions for if there
        # is/isn't a failing variant, once we've done enough runs to satisfy our required confidence level.
        # While min_early_exit_runs is technically calculated only for preventing false negatives, in practice it is
        # always less than max_success_runs (the anti-false-positive run threshold), so we will use it as a pre-req for
        # both checks. This ensures that we don't calculate max_success_runs until we have an accurate count of how many
        # molecules the solution uses at minimum (`N`).
        if num_runs >= min_early_exit_runs:
            # Exit early if there are no failing runs containing a molecule variant we haven't seen succeed yet
            # (even if not all variants have been seen).
            # It is sufficient for this to be true in either the restricted or unrestricted seed datasets
            if (all(n >= len(fail_run_variants[i])  # fail_run_variants isn't guaranteed to be of length N
                   or not (fail_run_variants[i][n] - success_run_variants[i][n])
                   for i, N in enumerate(Ns)
                   for n in range(1, N))
                or all(n >= len(fail_run_variants_first_match[i])
                       or not (fail_run_variants_first_match[i][n] - success_run_variants_first_match[i][n])
                       for i, N in enumerate(Ns)
                       for n in range(1, N))):
                if skip_non_precog_checks:
                    return None

                if verbose:
                    print(f"Solution is not precognitive; no failing variants found for {sum(Ns)} input molecules"
                          f" ({num_passing_runs} / {num_runs} runs passed, or"
                          f" {num_passing_runs_first_match} / {num_runs_first_match} for seeds with same first molecule).")

                return False

            if max_success_runs is None:
                max_success_runs = calc_max_success_runs(Ns, false_pos_rate=false_pos_rate)

            # Since we just confirmed that at least one missing variant is failing (in each seeds dataset), if we've
            # exceeded our max successful runs (i.e. we're confident we aren't false-positiving), mark the solution as
            # precog
            if num_passing_runs >= max_success_runs:
                if verbose or stderr_on_precog:
                    # This is redundant but I want to report which molecule was precognitive
                    for i, N in enumerate(Ns):
                        for n in range(1, N):
                            # Check for any variants of the ith input that appeared in a failing run but never in a succeeding run
                            if n < len(fail_run_variants[i]) and fail_run_variants[i][n] - success_run_variants[i][n]:
                                print(f"Solution is precognitive; molecule {n + 1} / {N} appears to always fail for some variants"
                                      f" ({num_passing_runs} / {num_runs} runs passed, or"
                                      f" {num_passing_runs_first_match} / {num_runs_first_match} for seeds with same first molecule).",
                                      file=STDERR if stderr_on_precog else STDOUT)
                                return True

                return True

        return None

    while total_cycles < max_total_cycles:
        if num_runs != 0:  # Smelly if, but this way `continue` is safe to use anywhere below
            for random_input in random_inputs:
                random_input.seed += 1

        solution.reset()  # Reset the solution from any prior run (this also picks up seed changes)

        # Check if the first molecule of each random input is the same as in the original input
        first_molecule_matches = all(random_input.get_next_molecule_idx() == first_input_variants[i]
                                     for i, random_input in enumerate(random_inputs))
        # Reset the inputs after that molecule check
        for random_input in random_inputs:
            random_input.reset()

        # Run the solution with this seed of the input, checking if it succeeds (ignoring the exact score)
        try:
            # Run the solution
            # if just_run_cycle_count was given, skip the first run to save time
            cycles = just_run_cycle_count if (num_runs == 0 and just_run_cycle_count) \
                     else solution.run(max_cycles=max_cycles).cycles

            # Check how many molecules the solution consumed for each random input, and lower each N if possible
            # Note that if just_run_cycle_count was provided, we initialized N already and reset the solution,
            # so we skip in that case
            if not (num_runs == 0 and just_run_cycle_count):
                for i, random_input in enumerate(random_inputs):
                    # Ignore molecules that only made it into the pipe since their variant can't affect the solution
                    this_N = random_input.num_inputs - sum(1 for mol in random_input.out_pipe if mol is not None)
                    Ns[i] = min(Ns[i], this_N)

            target_variants_data = success_run_variants
            target_variants_data_first_match = success_run_variants_first_match
            num_variants_to_store = Ns  # Direct reference is safe since we only read from this

            min_passing_run_cycles = min(min_passing_run_cycles, cycles)
            num_passing_runs += 1
            if first_molecule_matches:
                num_passing_runs_first_match += 1
        except Exception as e:
            if num_runs == 0:
                # Not allowed to crash on the original seed, otherwise do nothing
                raise Exception(f"Error in base seed: {type(e)}: {e}")

            cycles = solution.cycle
            target_variants_data = fail_run_variants  # The data set that this run's variants should be added to
            target_variants_data_first_match = fail_run_variants_first_match
            # Make sure we don't store data on variants of molecules from after the solution crashed
            num_variants_to_store = [random_input.num_inputs - sum(1 for mol in random_input.out_pipe if mol is not None)
                                     for random_input in random_inputs]

        total_cycles += cycles
        num_runs += 1
        if first_molecule_matches:
            num_runs_first_match += 1

        # Track all nth input variants that appeared in this run for 2 <= n <= N
        datasets_to_update = ([target_variants_data, target_variants_data_first_match]
                              if first_molecule_matches else
                              [target_variants_data])
        for dataset in datasets_to_update:
            for random_input, variants_data, num_new_variants in zip(random_inputs, dataset, num_variants_to_store):
                random_input.reset().get_next_molecule_idx()  # Reset and skip past n = 0

                if num_new_variants > len(variants_data):
                    variants_data.extend([set() for _ in range(num_new_variants - len(variants_data))])

                for n in range(1, num_new_variants):
                    variants_data[n].add(random_input.get_next_molecule_idx())

        # To save on futile runs, check if our time constraints can allow for us to get a sufficiently confident answer
        # about the success rate, assuming all future runs are successes or all are failures.
        # If we have so few runs that we can guarantee even our fallback confidence level won't be met, we can
        # immediately timeout
        remaining_cycles = max_total_cycles - total_cycles
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
        if not (  # Check if pure failures can confirm a too-low success rate (for restricted seeds, as explained above)
                binom.cdf(num_passing_runs_first_match - 1, num_runs_first_match - 1 + max_remaining_failing_runs,
                          NON_PRECOG_MIN_PASS_RATE)
                < MAX_FALSE_POS_RATE
                # Check if pure successes can confirm a sufficiently high success rate (for either seed type)
                or binom.cdf(num_runs - num_passing_runs,
                             num_runs - 1 + max_remaining_passing_runs,
                             1 - NON_PRECOG_MIN_PASS_RATE)
                   < MAX_FALSE_NEG_RATE
                or binom.cdf(num_runs_first_match - num_passing_runs_first_match,
                             num_runs_first_match - 1 + max_remaining_passing_runs,
                             1 - NON_PRECOG_MIN_PASS_RATE)
                   < MAX_FALSE_NEG_RATE):
            raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                               f" too few runs to ascertain {100 * NON_PRECOG_MIN_PASS_RATE}% success rate requirement"
                               f" ({num_passing_runs} / {num_runs} runs passed, or"
                               f" {num_passing_runs_first_match} / {num_runs_first_match} for seeds with same first input).")

        if success_rate_too_low(false_pos_rate=PREFERRED_FALSE_POS_RATE):
            return True

        molecule_assumption_check_result = check_molecule_assumptions(
            min_early_exit_runs=min_early_exit_runs,
            # Skip non-precog exit conditions if we aren't confident in the success rate yet
            skip_non_precog_checks=not success_rate_okay(false_neg_rate=PREFERRED_FALSE_NEG_RATE),
            false_pos_rate=PREFERRED_FALSE_POS_RATE)

        if molecule_assumption_check_result is not None:
            return molecule_assumption_check_result

    # If we escaped the run loop without returning, we've been time-constrained in our number of runs.
    # Attempt to redo the precog check with our fallback relaxed confidence levels, and if even then we aren't
    # sufficiently confident in our answer, raise an error
    if verbose:
        print("Warning: Precog check terminated early due to time constraints; check accuracy may be reduced.")

    if success_rate_too_low(false_pos_rate=MAX_FALSE_POS_RATE):
        return True

    if not success_rate_okay(false_neg_rate=MAX_FALSE_NEG_RATE):
        raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                           f" success rate too near {100 * NON_PRECOG_MIN_PASS_RATE}% requirement"
                           f" ({num_passing_runs} / {num_runs} = {100 * num_passing_runs / num_runs:.1f}% runs passed).")

    max_success_runs = None  # Reset max_success_runs so the check will re-calculate it
    molecule_assumption_check_result = check_molecule_assumptions(min_early_exit_runs=fallback_min_early_exit_runs,
                                                                  false_pos_rate=MAX_FALSE_POS_RATE)
    if molecule_assumption_check_result is not None:
        return molecule_assumption_check_result

    if num_runs < fallback_min_early_exit_runs:
        raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                           f" {num_runs} / {fallback_min_early_exit_runs} required runs executed.")
    else:
        raise TimeoutError("Precog check could not be completed to sufficient confidence due to time constraints;"
                           f" {num_passing_runs} / {max_success_runs} required passing runs completed.")
