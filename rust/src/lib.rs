//! Deterministic event-time processing research kernels.

/// A temporal event with an integer timestamp and unique identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub timestamp: i64,
    pub id: u64,
}

/// Return events in deterministic event-time order.
/// Equal timestamps are ordered by identifier.
pub fn canonical_order(events: &[Event]) -> Vec<Event> {
    let mut ordered = events.to_vec();
    ordered.sort_unstable_by_key(|event| (event.timestamp, event.id));
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_order_is_permutation_invariant() {
        let a = Event { timestamp: 2, id: 1 };
        let b = Event { timestamp: 1, id: 3 };
        let c = Event { timestamp: 2, id: 0 };

        let first = canonical_order(&[a.clone(), b.clone(), c.clone()]);
        let second = canonical_order(&[c, a, b]);

        assert_eq!(first, second);
        assert_eq!(first[0].timestamp, 1);
        assert_eq!(first[1].id, 0);
    }
}
