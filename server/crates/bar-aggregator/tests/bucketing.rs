//! Sanity checks on the 10s bucketing math. The bucket-floor function
//! must round DOWN to the nearest 10_000ms boundary regardless of sign.

use market_domain::{bucket_floor, BAR_INTERVAL_MS};

#[test]
fn bucket_floor_aligns_to_10s() {
    assert_eq!(bucket_floor(0), 0);
    assert_eq!(bucket_floor(1), 0);
    assert_eq!(bucket_floor(9_999), 0);
    assert_eq!(bucket_floor(10_000), 10_000);
    assert_eq!(bucket_floor(10_001), 10_000);
    assert_eq!(bucket_floor(15_500), 10_000);
    assert_eq!(bucket_floor(19_999), 10_000);
    assert_eq!(bucket_floor(20_000), 20_000);
}

#[test]
fn interval_constant_is_10_seconds() {
    assert_eq!(BAR_INTERVAL_MS, 10_000);
}
