// GENERATED FILE -- DO NOT EDIT.
//
// Written by frontend/scripts/generate-enums.py, which reads these values from
// the same Python modules serving/schema.py imports them from
// (src/data_validation.py, src/features.py). package.json regenerates this
// before `dev` and `build`, so it cannot drift from the API's real contract.
//
// Edit the Python sets, not this file.

/** src.data_validation.VALID_PURPOSE (14 values). Closed enum: an unseen
 *  purpose is a 422, never scored from LightGBM's untrained NaN bin. */
export const VALID_PURPOSE: readonly string[] = [
  "car",
  "credit_card",
  "debt_consolidation",
  "educational",
  "home_improvement",
  "house",
  "major_purchase",
  "medical",
  "moving",
  "other",
  "renewable_energy",
  "small_business",
  "vacation",
  "wedding",
] as const

/** src.data_validation.VALID_HOME_OWNERSHIP (4 values). */
export const VALID_HOME_OWNERSHIP: readonly string[] = [
  "MORTGAGE",
  "OTHER",
  "OWN",
  "RENT",
] as const

/** src.features.emp_order (11 values), IN ORDER -- this is an ordinal
 *  scale (add_features() maps it to emp_length_ord), so the order is meaning,
 *  not presentation. Do not sort. */
export const EMP_ORDER: readonly string[] = [
  "< 1 year",
  "1 year",
  "2 years",
  "3 years",
  "4 years",
  "5 years",
  "6 years",
  "7 years",
  "8 years",
  "9 years",
  "10+ years",
] as const

/** serving/schema.py's EMP_LENGTH_NOT_DISCLOSED. JSON null on emp_length MEANS
 *  "declined to disclose" and the backend normalizes it to exactly this string.
 *  It is a first-class choice, not a blank. */
export const EMP_LENGTH_NOT_DISCLOSED = "NI"

/** Numeric bounds ScoreRequest enforces (src/data_validation.py). Mirrored here
 *  so the form can refuse a value the API would 422 -- the API remains the
 *  authority; this is a courtesy, not a second gate. */
export const REVENUE_MIN = 1.0
export const LOAN_MIN = 500.0
export const LOAN_MAX = 40000.0
export const DTI_MAX_REAL = 1000.0
export const DTI_SENTINEL = 999.0
