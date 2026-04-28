//! Typed errors with stable codes (MSS micro-pattern §7.1).

use thiserror::Error;

/// Domain-level errors. Codes (E_XXX) match `agent/diagnostics.schema.json`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DomainError {
    #[error("E_INVALID_INSTRUMENT: expected ^[A-Z]{{3}}_[A-Z]{{3}}$, got {0:?}")]
    InvalidInstrument(String),

    #[error("E_INVALID_ACCOUNT_ID: must be non-empty, got empty")]
    EmptyAccountId,

    #[error("E_INVALID_ORDER_ID: must be non-empty, got empty")]
    EmptyOrderId,

    #[error("E_INVALID_TRANSACTION_ID: must be non-empty, got empty")]
    EmptyTransactionId,

    #[error("E_OVERFLOW: arithmetic overflow on Units")]
    UnitsOverflow,
}
