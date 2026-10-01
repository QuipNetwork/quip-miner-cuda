//! The kernel a process runs, and the one name each kernel carries.
//!
//! Three self-feeding kernels share the host: `kernels/sa.cu`,
//! `kernels/msa.cu` and `kernels/gibbs.cu`. Everything that varies by kernel
//! (capacity, read cap, blocks per nonce, launch arguments, identity) keys on
//! this enum. A fourth kernel is therefore a compile error at every site that
//! must know about it, not a silent fall-through.

use quip_solver_core::quip_proto::v1::Algorithm;

/// Which self-feeding kernel to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// Metropolis simulated annealing, one thread per read
    /// (`cuda_sa_self_feeding`).
    Sa,
    /// Multi-spin coded simulated annealing, 64 reads per word
    /// (`cuda_msa_self_feeding`).
    Msa,
    /// Single-site heat-bath Gibbs (`cuda_gibbs_self_feeding`).
    Gibbs,
}

impl KernelKind {
    /// The one name this kernel goes by: the wire `algorithm` name (see
    /// [`Self::algorithm`]), the binary suffix (`quip-cuda-<name>`), the kernel file stem and the JIT cache
    /// key.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sa => "sa",
            Self::Msa => "msa",
            Self::Gibbs => "gibbs",
        }
    }

    /// The wire `algorithm` this kernel advertises in Hello and
    /// `--capabilities`. The protocol renders it back to [`Self::name`].
    #[must_use]
    pub const fn algorithm(self) -> Algorithm {
        match self {
            Self::Sa => Algorithm::Sa,
            Self::Msa => Algorithm::Msa,
            Self::Gibbs => Algorithm::Gibbs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KernelKind;
    use quip_solver_core::quip_protocol::session::algorithm_name;

    /// The wire enum must render to the same name the binary and cache use,
    /// or `--capabilities` would advertise one kernel under another's name.
    #[test]
    fn wire_algorithm_renders_to_the_kernel_name() {
        for kind in [KernelKind::Sa, KernelKind::Msa, KernelKind::Gibbs] {
            assert_eq!(algorithm_name(kind.algorithm()), kind.name());
        }
    }

    /// The name is a key in three places (wire, binary, cache), so it must be
    /// distinct per kernel and free of characters a filename cannot carry.
    #[test]
    fn names_are_distinct_lowercase_ascii() {
        let names = [
            KernelKind::Sa.name(),
            KernelKind::Msa.name(),
            KernelKind::Gibbs.name(),
        ];
        assert_eq!(names, ["sa", "msa", "gibbs"]);
        for name in names {
            assert!(name.bytes().all(|b| b.is_ascii_lowercase()), "{name}");
        }
    }
}
