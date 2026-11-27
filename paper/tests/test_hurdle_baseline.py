"""Hurdle model: the shuffle-null baselines that enter as offsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import stats_bayesian_hurdle_glmm_chirprate_deviation_during_chasing_offsets as hurdle


def test_conditional_null_mean_is_the_mean_over_positive_permutations():
    mu = hurdle.conditional_null_mean(np.array([6.0, 3.0]), np.array([4, 3]))
    np.testing.assert_allclose(mu, [1.5, 1.0])


def test_conditional_null_mean_is_one_where_no_permutation_has_a_chirp():
    # was 0, which entered the positive-count model as log(1e-6) = -13.8
    mu = hurdle.conditional_null_mean(np.array([0.0, 5.0]), np.array([0, 2]))
    np.testing.assert_allclose(mu, [1.0, 2.5])
    assert np.all(np.isfinite(np.log(mu)))


@pytest.mark.parametrize(
    ("pos_sum", "pos_count"),
    [([2.0], [0]),  # a count sum without positive permutations
     ([1.0], [2]),  # mean count below 1 among positive permutations
     ([1.0], [-1])],
)
def test_conditional_null_mean_refuses_inconsistent_tallies(pos_sum, pos_count):
    with pytest.raises(ValueError):
        hurdle.conditional_null_mean(np.array(pos_sum), np.array(pos_count))


def _wide(counts_loser, counts_winner):
    n = len(counts_loser)
    return pd.DataFrame({
        "recording": ["r1"] * n,
        "loser_id": [1] * n, "winner_id": [2] * n,
        "loser_sex": ["f"] * n, "winner_sex": ["m"] * n,
        "n_loser_chirps_offset_window": counts_loser,
        "n_winner_chirps_offset_window": counts_winner,
    })


def test_shuffle_baselines_of_a_window_never_hit_by_the_null(tmp_path):
    resp = "n_chirps_offset_window"
    obs = tmp_path / "obs.csv"
    _wide([3, 0], [1, 0]).to_csv(obs, index=False)
    perms = []
    # Rows after stacking: loser chase 0, loser chase 1, winner chase 0, winner chase 1.
    # The winner of chase 0 chirped once, but no permutation put a chirp in its window.
    for b, (l0, w0) in enumerate([(2, 0), (0, 0), (4, 0), (1, 0)]):
        p = tmp_path / f"perm_{b}.csv"
        _wide([l0, 0], [w0, 0]).to_csv(p, index=False)
        perms.append(p)
    df = hurdle.compute_shuffle_baselines(
        hurdle.load_data_with_interaction_id(obs), perms, resp, log_every=0)
    np.testing.assert_array_equal(df["n_perm_pos"], [3, 0, 0, 0])
    np.testing.assert_allclose(df["mu_null_pos"], [7 / 3, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(df["p_null_pos"], [3.5 / 5, 0.5 / 5, 0.5 / 5, 0.5 / 5])
    assert df.loc[(df["status"] == 1) & (df[resp] > 0), "mu_null_pos"].item() == 1.0
