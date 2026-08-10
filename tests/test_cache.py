import numpy as np

from core.types.cache import ProcessingCache


def test_cache_requires_complete_products():
    cache = ProcessingCache(pga=1.0, pgv=2.0, pgd=3.0)
    assert not cache.has_strong_motion_parameters
    cache.arias_intensity = 1.0
    cache.cumulative_absolute_velocity = 2.0
    cache.significant_duration_5_75 = 3.0
    cache.significant_duration_5_95 = 4.0
    assert cache.has_strong_motion_parameters


def test_cache_invalidation_clears_transitive_products():
    cache = ProcessingCache(
        velocity=np.ones(3), displacement=np.ones(3),
        response_periods=np.array([0.1]), spectral_displacement=np.ones(1),
        spectral_velocity=np.ones(1), spectral_acceleration=np.ones(1),
        pseudo_spectral_velocity=np.ones(1), pseudo_spectral_acceleration=np.ones(1),
    )
    assert cache.has_time_domain and cache.has_response_spectrum
    cache.clear_time_domain()
    assert not cache.has_time_domain
    assert not cache.has_response_spectrum
