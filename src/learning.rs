//! Shared online learning primitives for caller-owned linear rank models.

pub const DEFAULT_LEARNING_RATE: f64 = 0.05;
pub const DEFAULT_MARGIN: f64 = 1.0;
pub const MAX_WEIGHT: f64 = 10_000.0;

/// Apply one pairwise perceptron-style update to a fixed-size weight vector.
///
/// Bias cancels when comparing two candidates, so the update only needs the
/// feature difference. Values are clipped before updating to keep a malformed
/// or unusually large caller feature from destabilising the model.
pub fn learn_pair<const N: usize>(
    weights: &mut [f64; N],
    preferred: &[f64; N],
    other: &[f64; N],
    learning_rate: f64,
    margin: f64,
) -> bool {
    let learning_rate = finite(learning_rate).max(0.0);
    let margin = finite(margin).max(0.0);
    let difference = weights
        .iter()
        .zip(preferred)
        .zip(other)
        .map(|((weight, preferred), other)| finite(*weight) * (finite(*preferred) - finite(*other)))
        .sum::<f64>();
    if difference >= margin || learning_rate == 0.0 {
        return false;
    }

    let mut changed = false;
    for ((weight, preferred), other) in weights.iter_mut().zip(preferred).zip(other) {
        let delta = (finite(*preferred) - finite(*other)).clamp(-1000.0, 1000.0);
        if delta != 0.0 {
            changed = true;
        }
        *weight = (finite(*weight) + learning_rate * delta).clamp(-MAX_WEIGHT, MAX_WEIGHT);
    }
    changed
}

fn finite(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::learn_pair;

    #[test]
    fn pairwise_update_moves_the_preferred_vector_up() {
        let mut weights = [0.0; 2];
        let preferred = [1.0, 0.0];
        let other = [0.0, 1.0];

        assert!(learn_pair(&mut weights, &preferred, &other, 1.0, 1.0));
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn identical_vectors_are_a_noop() {
        let mut weights = [1.0; 2];
        let before = weights;
        assert!(!learn_pair(
            &mut weights,
            &[2.0, 3.0],
            &[2.0, 3.0],
            1.0,
            1.0
        ));
        assert_eq!(weights, before);
    }
}
