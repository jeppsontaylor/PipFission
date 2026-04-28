//! Typed IDs (MSS micro-pattern §7.1: typed IDs + validated constructors).
//!
//! All IDs are `#[serde(transparent)]` so they remain wire-compatible with
//! the dashboard's existing `string`/`number` JSON shapes.

use serde::{Deserialize, Serialize};

use crate::errors::DomainError;

// ---------- Instrument ----------

/// FX instrument symbol. OANDA format: `[A-Z]{3}_[A-Z]{3}`, e.g. `EUR_USD`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Instrument(String);

impl Instrument {
    /// Validated constructor. Accepts only `[A-Z]{3}_[A-Z]{3}` shape.
    pub fn try_new(s: impl AsRef<str>) -> Result<Self, DomainError> {
        let s = s.as_ref();
        if Self::is_valid(s) {
            Ok(Self(s.to_owned()))
        } else {
            Err(DomainError::InvalidInstrument(s.to_owned()))
        }
    }

    /// Parses without bubbling the error — useful where the caller treats
    /// invalid symbols as a skipped row (e.g. malformed pricing line).
    pub fn parse(s: impl AsRef<str>) -> Option<Self> {
        Self::try_new(s).ok()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }

    fn is_valid(s: &str) -> bool {
        // [A-Z]{3}_[A-Z]{3}
        s.len() == 7
            && s.is_ascii()
            && s.as_bytes()[3] == b'_'
            && s.as_bytes()[..3].iter().all(|b| b.is_ascii_uppercase())
            && s.as_bytes()[4..].iter().all(|b| b.is_ascii_uppercase())
    }
}

impl AsRef<str> for Instrument {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Instrument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------- AccountId ----------

/// OANDA account identifier (e.g. `001-001-1234567-001`). Non-empty.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccountId(String);

impl AccountId {
    pub fn try_new(s: impl Into<String>) -> Result<Self, DomainError> {
        let s = s.into();
        if s.is_empty() {
            Err(DomainError::EmptyAccountId)
        } else {
            Ok(Self(s))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for AccountId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AccountId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------- OrderId / TransactionId ----------

/// OANDA order ID. Numeric on the wire, but kept as a string for forward
/// compatibility with internal paper-book IDs (which we generate as
/// `paper-N`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OrderId(String);

impl OrderId {
    pub fn try_new(s: impl Into<String>) -> Result<Self, DomainError> {
        let s = s.into();
        if s.is_empty() {
            Err(DomainError::EmptyOrderId)
        } else {
            Ok(Self(s))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for OrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TransactionId(String);

impl TransactionId {
    pub fn try_new(s: impl Into<String>) -> Result<Self, DomainError> {
        let s = s.into();
        if s.is_empty() {
            Err(DomainError::EmptyTransactionId)
        } else {
            Ok(Self(s))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------- Units ----------

/// Signed order/position units (long > 0, short < 0). i64 on the wire.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Units(pub i64);

impl Units {
    pub const ZERO: Self = Self(0);

    pub fn checked_add(self, rhs: Self) -> Result<Self, DomainError> {
        self.0
            .checked_add(rhs.0)
            .map(Self)
            .ok_or(DomainError::UnitsOverflow)
    }

    pub fn checked_sub(self, rhs: Self) -> Result<Self, DomainError> {
        self.0
            .checked_sub(rhs.0)
            .map(Self)
            .ok_or(DomainError::UnitsOverflow)
    }

    pub fn signum(self) -> i64 {
        self.0.signum()
    }

    pub fn abs(self) -> u64 {
        self.0.unsigned_abs()
    }
}

impl std::fmt::Display for Units {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn instrument_accepts_valid() {
        for s in ["EUR_USD", "USD_JPY", "XAU_USD", "ABC_DEF"] {
            assert!(Instrument::try_new(s).is_ok(), "should accept {s}");
        }
    }

    #[test]
    fn instrument_rejects_invalid() {
        for s in [
            "",
            "EURUSD",
            "eur_usd",
            "EUR-USD",
            "EUR_USDX",
            "EU_USD",
            "EURR_USD",
            "EUR_USDD",
            "1EU_USD",
            "EUR_US1",
            "EUR_USDX_X",
        ] {
            assert!(Instrument::try_new(s).is_err(), "should reject {s:?}");
        }
    }

    #[test]
    fn instrument_serializes_transparently() {
        let inst = Instrument::try_new("EUR_USD").unwrap();
        let json = serde_json::to_string(&inst).unwrap();
        assert_eq!(json, r#""EUR_USD""#);
        let back: Instrument = serde_json::from_str(&json).unwrap();
        assert_eq!(back, inst);
    }

    #[test]
    fn account_id_rejects_empty() {
        assert_eq!(AccountId::try_new(""), Err(DomainError::EmptyAccountId));
        assert!(AccountId::try_new("001-001-1234567-001").is_ok());
    }

    #[test]
    fn units_arithmetic_does_not_overflow() {
        let a = Units(i64::MAX);
        let b = Units(1);
        assert!(matches!(a.checked_add(b), Err(DomainError::UnitsOverflow)));
        let c = Units(i64::MIN);
        assert!(matches!(c.checked_sub(b), Err(DomainError::UnitsOverflow)));
    }

    proptest! {
        #[test]
        fn instrument_validation_total(s in ".*") {
            // Should never panic; either Ok or Err for any string.
            let _ = Instrument::try_new(&s);
        }

        #[test]
        fn instrument_roundtrip_via_json(
            l in "[A-Z]{3}",
            r in "[A-Z]{3}"
        ) {
            let s = format!("{l}_{r}");
            let inst = Instrument::try_new(&s).unwrap();
            let json = serde_json::to_string(&inst).unwrap();
            let back: Instrument = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(inst, back);
        }

        #[test]
        fn units_checked_add_matches_i64(a: i64, b: i64) {
            let lhs = Units(a).checked_add(Units(b));
            match a.checked_add(b) {
                Some(sum) => prop_assert_eq!(lhs, Ok(Units(sum))),
                None => prop_assert!(matches!(lhs, Err(DomainError::UnitsOverflow))),
            }
        }
    }
}
