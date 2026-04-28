//! Exponential backoff with cap. Simple, stateful-on-the-stack.

use std::time::Duration;

pub struct Backoff {
    current_ms: u64,
    cap_ms: u64,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Backoff {
    pub fn new() -> Self {
        Self {
            current_ms: 250,
            cap_ms: 30_000,
        }
    }

    pub fn next_delay(&mut self) -> Duration {
        let d = Duration::from_millis(self.current_ms);
        self.current_ms = self.current_ms.saturating_mul(2).min(self.cap_ms);
        d
    }

    pub fn reset(&mut self) {
        self.current_ms = 250;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_is_250ms() {
        let mut b = Backoff::new();
        assert_eq!(b.next_delay(), Duration::from_millis(250));
    }

    #[test]
    fn doubles_until_cap() {
        let mut b = Backoff::new();
        let mut last = b.next_delay();
        for _ in 0..20 {
            let next = b.next_delay();
            assert!(next >= last, "non-monotonic");
            assert!(next.as_millis() <= 30_000, "exceeded cap");
            last = next;
        }
        // Eventually pinned at 30s.
        assert_eq!(b.next_delay(), Duration::from_millis(30_000));
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut b = Backoff::new();
        for _ in 0..5 {
            b.next_delay();
        }
        b.reset();
        assert_eq!(b.next_delay(), Duration::from_millis(250));
    }
}
