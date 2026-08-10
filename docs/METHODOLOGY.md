# BSMA processing methodology

## Scope and input declaration

Each run is declared as either **raw instrument counts** (requires a matching
StationXML response) or **processed physical acceleration** (requires an
operator-declared unit). BSMA does not infer either condition from a MiniSEED
extension or filename alone. A label such as `BP4` is evidence that a prior
filter may have been used, not proof of unit or calibration.

## Processing chain

1. Validate samples, sampling rate, continuity, clipping, flatline segments,
   and impulsive anomalies. Critical faults halt processing; warnings remain in
   the audit trail.
2. For raw counts only, remove the matched instrument response to m/s2.
3. Correct the baseline, apply a Tukey taper, and apply a zero-phase
   Butterworth filter. Default corners (0.25-25 Hz) are a starting point and
   must be reviewed against record SNR and Nyquist frequency.
4. Integrate acceleration using the cumulative trapezoid rule to obtain
   velocity and displacement.
5. Compute PGA, PGV, PGD, Arias intensity, Husid energy curve, D5-D95, and
   5%-damped response spectra.

## Validation and limitations

Numerical validation is dataset-specific. A benchmark compares BSMA against a
reference result only when the same event window, response treatment, unit,
filter, and sampling are documented. A mismatch is reported, never silently
normalized. Results are unsuitable for engineering decisions until the
operator has reviewed provenance, QC, filter corners, and benchmark status.

## References

- USGS PRISM, *Strong-Motion Data Processing* (OFR 2017-1008).
- PEER NGA-West2, Ancheta et al. (2013), strong-motion processing procedure.
- BMKG SIG-BMKG intensity thresholds supplied with the project review.
