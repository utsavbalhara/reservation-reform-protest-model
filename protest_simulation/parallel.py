"""Run many independent campaigns across processes.

Each job carries its own parameters and campaign seed, so the results are identical to running the same jobs
serially, in any order and with any number of workers. The population is sent to each worker once.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .protest_campaign import simulate_protest_campaign

_WORKER_POPULATION = None


def default_worker_count() -> int:
    configured = os.environ.get("PROTEST_WORKERS")
    if configured:
        return max(1, int(configured))
    return max(1, os.cpu_count() or 1)


def _install_population(population):
    global _WORKER_POPULATION
    _WORKER_POPULATION = population


def _run_job(job):
    parameters, campaign_seed, record_neighbourhood_turnout, record_agent_protest_days = job
    return simulate_protest_campaign(_WORKER_POPULATION, parameters, np.random.default_rng(campaign_seed),
                                     record_neighbourhood_turnout=record_neighbourhood_turnout,
                                     record_agent_protest_days=record_agent_protest_days)


def run_campaigns(population, jobs, workers: int = None):
    """jobs: iterable of (parameters, campaign_seed), optionally followed by record_neighbourhood_turnout and
    record_agent_protest_days flags."""
    normalised = [tuple(job) + (False,) * (4 - len(job)) for job in jobs]
    workers = default_worker_count() if workers is None else workers
    if workers <= 1 or len(normalised) <= 1:
        _install_population(population)
        return [_run_job(job) for job in normalised]
    chunk = max(1, len(normalised) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers, initializer=_install_population, initargs=(population,)) as pool:
        return list(pool.map(_run_job, normalised, chunksize=chunk))
