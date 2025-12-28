# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class Resampler:
    """
    Scientifically correct physics-based resampler.

    Principles:
    - Physics is generated ONLY at latent_freq (e.g. 10 min)
    - Physics is initialized using real measurements (as-of backward)
    - Original timestamps are preserved exactly
    - No interpolation is performed
    - Generated vs observed rows are explicitly flagged
    - Each well is treated independently
    """

    def __init__(
        self,
        base_freq: str = "30s",
        latent_freq: str = "10min",
        choke_step_hours: int = 6,
        seed: int | None = None,
    ):
        self.base_freq = base_freq
        self.latent_freq = latent_freq
        self.choke_step_hours = choke_step_hours
        self.base_seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resample_wells(
        self,
        df: pd.DataFrame,
        well_id_col: str = "well_id",
        well_code_col: str = "well_code",
        independent_vars: list[str] = None,
    ) -> pd.DataFrame:
        if independent_vars is None:
            raise ValueError("independent_vars must be provided")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")

        parts = []

        for i, (well_id, df_well) in enumerate(df.groupby(well_id_col)):
            rng = np.random.default_rng(
                None if self.base_seed is None else self.base_seed + i
            )

            well_codes = df_well[well_code_col].unique()
            if len(well_codes) != 1:
                raise ValueError("Multiple well_code values for same well_id")

            well_code = well_codes[0]

            df_resampled = self._resample_well(
                df_well=df_well.drop(columns=[well_id_col]),
                rng=rng,
                independent_vars=independent_vars,
            )

            df_resampled[well_id_col] = well_id
            df_resampled[well_code_col] = well_code
            parts.append(df_resampled)

        df_concatented = pd.concat(parts).sort_index()
        t0 = df_concatented.index.min()
        delta = (df_concatented.index - t0) / pd.Timedelta(self.base_freq)
        df_concatented["time_idx"] = delta.astype(int)
        df_concatented = df_concatented.sort_values(["well_id", "time_idx"]).reset_index(drop=True)
        return df_concatented

    # ------------------------------------------------------------------
    # Single well logic
    # ------------------------------------------------------------------

    def _resample_well(
        self,
        df_well: pd.DataFrame,
        rng: np.random.Generator,
        independent_vars: list[str],
    ) -> pd.DataFrame:

        df_well = df_well.sort_index()
        original_index = df_well.index

        df_measured_indep = df_well[independent_vars]
        df_other = df_well.drop(columns=independent_vars)

        # ------------------------------------------------------------------
        # Physics start time: FIRST VALID MEASUREMENT, CEILED
        # ------------------------------------------------------------------
        first_obs = df_measured_indep.dropna(how="all").index.min()
        if pd.isna(first_obs):
            raise ValueError("No independent-variable measurements available")

        start = first_obs.ceil(self.latent_freq)
        end = original_index.max().ceil(self.latent_freq)

        latent_index = pd.date_range(
            start=start,
            end=end,
            freq=self.latent_freq,
            tz=original_index.tz,
        )

        # ------------------------------------------------------------------
        # AS-OF backward anchoring (CRITICAL)
        # ------------------------------------------------------------------
        latent_base = pd.merge_asof(
            pd.DataFrame(index=latent_index),
            df_measured_indep.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )

        # ------------------------------------------------------------------
        # Generate latent physics
        # ------------------------------------------------------------------
        latent_vars = self._generate_latent_variables(
            baseline=latent_base,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # Union of original + physics timestamps
        # ------------------------------------------------------------------
        final_index = original_index.union(latent_index).sort_values()
        latent_vars = latent_vars.reindex(final_index)

        # ------------------------------------------------------------------
        # Overwrite ONLY observed timestamps
        # ------------------------------------------------------------------
        latent_vars.loc[original_index, independent_vars] = (
            df_measured_indep.loc[original_index]
        )

        df_other = df_other.reindex(final_index)

        is_observed = final_index.isin(original_index).astype(int)

        df_out = pd.concat([latent_vars, df_other], axis=1)
        df_out["is_observed"] = is_observed

        return df_out

    # ------------------------------------------------------------------
    # Latent physics generation
    # ------------------------------------------------------------------

    def _generate_latent_variables(
        self,
        baseline: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:

        choke = self._generate_choke(baseline["choke"], rng)

        whp = self._pressure_response(
            baseline["whp"].values, choke, 0.25, 0.08, 0.05, rng
        )
        dcp = self._pressure_response(
            baseline["dcp"].values, choke, 0.15, 0.08, 0.04, rng
        )
        dhp = self._pressure_response(
            baseline["dhp"].values, choke, 0.05, 0.03, 0.02, rng
        )

        wht = self._thermal_response(baseline["wht"].values, rng)
        dht = self._thermal_response(baseline["dht"].values, rng)

        return pd.DataFrame(
            {
                "choke": choke,
                "whp": whp,
                "dcp": dcp,
                "dhp": dhp,
                "wht": wht,
                "dht": dht,
            },
            index=baseline.index,
        )

    # ------------------------------------------------------------------
    # Physics models
    # ------------------------------------------------------------------

    def _generate_choke(self, base: pd.Series, rng) -> np.ndarray:
        choke = base.values.copy()
        step = int(self.choke_step_hours * 60 / self._latent_minutes())
        step = max(step, 1)

        for i in range(0, len(choke), step):
            choke[i:] += rng.choice([-2, 0, 2])

        return np.clip(choke, 5, 100)

    def _pressure_response(
        self,
        base: np.ndarray,
        choke: np.ndarray,
        sensitivity: float,
        relax_10min: float,
        noise_10min: float,
        rng,
    ) -> np.ndarray:

        relax = relax_10min
        noise = noise_10min

        p = np.full(len(base), np.nan)

        valid = np.flatnonzero(~np.isnan(base))
        if len(valid) == 0:
            return p

        p[valid[0]] = base[valid[0]]

        for t in range(valid[0] + 1, len(base)):
            target = base[t] - sensitivity * (choke[t] - choke[valid[0]])
            p[t] = (
                p[t - 1]
                + relax * (target - p[t - 1])
                + noise * rng.normal()
            )

        return p

    def _thermal_response(
        self,
        base: np.ndarray,
        rng,
        noise_10min: float = 0.003,
    ) -> np.ndarray:

        noise = noise_10min
        drift = np.cumsum(rng.normal(size=len(base))) * noise
        return base + drift

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _latent_minutes(self) -> int:
        return int(pd.Timedelta(self.latent_freq).total_seconds() / 60)
