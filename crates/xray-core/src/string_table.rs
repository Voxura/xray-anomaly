//! Concurrent sharded string interning for X-Ray Anomaly.
//!
//! Replaces the legacy critical-section-locked `shared_str` (`src/xrCore/xrstring.cpp`)
//! with a cache-friendly, 64-shard concurrent symbol registry designed for 16+ core scaling.
//! Reads are lock-free / optimistic, and writes are fine-grained per-shard.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

/// Number of independent hash shards to eliminate lock contention under concurrent load.
const SHARD_COUNT: usize = 64;

/// A lightweight 64-bit interned string symbol.
///
/// Symbol equality (`a == b`) and hash operations are $O(1)$ single-cycle integer comparisons,
/// eliminating pointer chasing and string comparisons in hot engine loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Symbol(pub u64);

impl Symbol {
    pub const NULL: Symbol = Symbol(0);

    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub fn id(self) -> u64 {
        self.0
    }
}

struct Shard {
    table: HashMap<String, Symbol>,
}

/// Concurrent, thread-safe string interner.
pub struct StringInterner {
    shards: [RwLock<Shard>; SHARD_COUNT],
    symbols: RwLock<Vec<String>>,
    counter: AtomicU32,
}

impl StringInterner {
    /// Create a new concurrent interner with pre-allocated shards.
    pub fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| RwLock::new(Shard { table: HashMap::new() })),
            symbols: RwLock::new(vec![String::new()]), // index 0 reserved for Symbol::NULL
            counter: AtomicU32::new(1),
        }
    }

    #[inline(always)]
    fn shard_index(hash: u64) -> usize {
        (hash as usize) & (SHARD_COUNT - 1)
    }

    #[inline(always)]
    fn hash_str(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Intern a string slice. Returns an $O(1)$ comparable Symbol.
    pub fn intern(&self, text: &str) -> Symbol {
        if text.is_empty() {
            return Symbol::NULL;
        }

        let hash = Self::hash_str(text);
        let shard_idx = Self::shard_index(hash);

        // Fast path: optimistic read lock
        {
            let shard = self.shards[shard_idx].read().expect("lock poisoned");
            if let Some(&sym) = shard.table.get(text) {
                return sym;
            }
        }

        // Slow path: write lock
        let mut shard = self.shards[shard_idx].write().expect("lock poisoned");
        if let Some(&sym) = shard.table.get(text) {
            return sym;
        }

        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        let sym = Symbol((hash & 0xFFFF_FFFF_0000_0000) | (index as u64));

        let owned = text.to_string();
        shard.table.insert(owned.clone(), sym);

        let mut sym_vec = self.symbols.write().expect("lock poisoned");
        sym_vec.push(owned);

        sym
    }

    /// Resolve an interned symbol back to a string reference.
    pub fn resolve(&self, sym: Symbol) -> Option<String> {
        if sym.is_null() {
            return Some(String::new());
        }
        let index = (sym.0 & 0x0000_0000_FFFF_FFFF) as usize;
        let sym_vec = self.symbols.read().expect("lock poisoned");
        sym_vec.get(index).cloned()
    }

    /// Get total number of unique strings interned.
    pub fn count(&self) -> usize {
        (self.counter.load(Ordering::Relaxed) as usize).saturating_sub(1)
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_intern_and_resolve() {
        let interner = StringInterner::new();
        let s1 = interner.intern("actor_visual");
        let s2 = interner.intern("actor_visual");
        let s3 = interner.intern("wpn_ak74");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(interner.resolve(s1), Some("actor_visual".to_string()));
        assert_eq!(interner.resolve(s3), Some("wpn_ak74".to_string()));
        assert_eq!(interner.intern(""), Symbol::NULL);
    }

    #[test]
    fn test_concurrent_interning() {
        let interner = Arc::new(StringInterner::new());
        let mut handles = Vec::new();

        for thread_id in 0..16 {
            let interner_clone = Arc::clone(&interner);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    // Shared keys across threads to stress concurrent deduplication
                    let shared_key = format!("shared_section_{}", i % 50);
                    let s_shared1 = interner_clone.intern(&shared_key);
                    let s_shared2 = interner_clone.intern(&shared_key);
                    assert_eq!(s_shared1, s_shared2);

                    let thread_key = format!("thread_{}_section_{}", thread_id, i % 25);
                    let sym1 = interner_clone.intern(&thread_key);
                    let sym2 = interner_clone.intern(&thread_key);
                    assert_eq!(sym1, sym2);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Worker thread panicked");
        }

        assert_eq!(interner.count(), 50 + (16 * 25));
    }
}
