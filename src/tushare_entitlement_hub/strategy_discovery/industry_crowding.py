"""Prepare auditable industry inputs without replacing the domain percentile definition."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def complete_industry_amount_shares(amounts: FloatArray) -> FloatArray:
    """Divide each date by its complete, same-level public industry denominator.

    The caller supplies the historically applicable catalog as the column axis;
    unknown catalog dates must contain NaN. Missing, infinite or negative amounts
    invalidate the whole date, never silently shrinking the denominator. All-zero
    dates are unknown. Input [[1, 3], [1, NaN]] returns [[.25, .75], [NaN, NaN]].
    No provider units are changed because numerator and denominator cancel.
    """
    if amounts.ndim != 2 or amounts.shape[1] == 0:
        raise ValueError("industry amounts require dates by nonempty catalog")
    output = np.full(amounts.shape, np.nan)
    complete = np.all(np.isfinite(amounts) & (amounts >= 0), axis=1)
    totals = np.sum(np.where(complete[:, None], amounts, 0), axis=1)
    usable = complete & np.isfinite(totals) & (totals > 0)
    output[usable] = amounts[usable] / totals[usable, None]
    return output


def candidate_crowding_features(
    percentiles: FloatArray,
    sector_indices: FloatArray,
    candidate_days: NDArray[np.int64],
    candidate_securities: NDArray[np.int64],
    *,
    change_sessions: int = 5,
) -> FloatArray:
    """Map approved domain percentiles to candidate rows using dated membership.

    Input percentiles has date/industry axes; sector_indices has date/security
    axes. Unknown memberships stay NaN. A five-session change is known only when
    all six percentiles exist and the security stays in the same industry.
    Example: a stock in industry 0 at every date with percentile .2 then .7 five
    sessions later receives [.7, .5]. Neither future membership nor future data
    is consulted. The caller gates pre-publication and unqualified history dates.
    """
    if (
        percentiles.ndim != 2
        or percentiles.shape[1] == 0
        or sector_indices.ndim != 2
        or len(percentiles) != len(sector_indices)
        or candidate_days.ndim != 1
        or candidate_securities.shape != candidate_days.shape
        or not np.issubdtype(candidate_days.dtype, np.integer)
        or not np.issubdtype(candidate_securities.dtype, np.integer)
        or change_sessions < 1
        or np.any(candidate_days < 0)
        or np.any(candidate_days >= len(percentiles))
        or np.any(candidate_securities < 0)
        or np.any(candidate_securities >= sector_indices.shape[1])
        or np.any(np.isinf(percentiles))
        or np.any(np.isfinite(percentiles) & ((percentiles < 0) | (percentiles > 1)))
    ):
        raise ValueError("invalid crowding axes or percentile values")
    output = np.full((len(candidate_days), 2), np.nan)
    sectors = sector_indices[candidate_days, candidate_securities]
    valid = (
        np.isfinite(sectors)
        & (sectors >= 0)
        & (sectors < percentiles.shape[1])
        & (sectors == np.floor(sectors))
    )
    rows = np.flatnonzero(valid)
    selected_sectors = sectors[rows].astype(int)
    output[rows, 0] = percentiles[candidate_days[rows], selected_sectors]
    change_rows = rows[candidate_days[rows] >= change_sessions]
    stable = np.ones(len(change_rows), dtype=bool)
    change_sectors = sectors[change_rows].astype(int)
    for offset in range(change_sessions + 1):
        days = candidate_days[change_rows] - offset
        stable &= sector_indices[days, candidate_securities[change_rows]] == change_sectors
        stable &= np.isfinite(percentiles[days, change_sectors])
    change_rows = change_rows[stable]
    output[change_rows, 1] = (
        output[change_rows, 0]
        - percentiles[
            candidate_days[change_rows] - change_sessions, sectors[change_rows].astype(int)
        ]
    )
    return output
