//! Precreated session streams, checked out as indivisible pairs.

use cudarc::driver::{CudaContext, CudaStream, DriverError};
use std::sync::{Arc, Condvar, Mutex};

// Covers the 48-session QUI-1598 reproducer with spare capacity. Exhaustion
// blocks checkout, capping concurrent sessions per CudaDevice at 64. Creating
// streams lazily can deadlock the driver while persistent kernels are running.
const SESSION_STREAM_PAIRS: usize = 64;

struct Pool<T> {
    available: Mutex<Vec<T>>,
    returned: Condvar,
}

impl<T> Pool<T> {
    fn new(items: Vec<T>) -> Self {
        Self {
            available: Mutex::new(items),
            returned: Condvar::new(),
        }
    }

    fn take(&self) -> T {
        let mut available = self.available.lock().unwrap();
        loop {
            if let Some(item) = available.pop() {
                return item;
            }
            available = self.returned.wait(available).unwrap();
        }
    }

    fn put(&self, item: T) {
        self.available.lock().unwrap().push(item);
        self.returned.notify_one();
    }
}

type StreamPair = (Arc<CudaStream>, Arc<CudaStream>);

pub(crate) struct StreamPool {
    pairs: Pool<StreamPair>,
}

impl StreamPool {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let mut pairs = Vec::with_capacity(SESSION_STREAM_PAIRS);
        for _ in 0..SESSION_STREAM_PAIRS {
            pairs.push((ctx.new_stream()?, ctx.new_stream()?));
        }
        Ok(Self {
            pairs: Pool::new(pairs),
        })
    }

    pub(crate) fn take(&self) -> SessionStreams<'_> {
        let (compute, transfer) = self.pairs.take();
        SessionStreams {
            compute,
            transfer,
            pool: &self.pairs,
        }
    }
}

/// Keep this lease after the session's buffers so their queued frees drain too.
pub(crate) struct SessionStreams<'a> {
    pub(crate) compute: Arc<CudaStream>,
    pub(crate) transfer: Arc<CudaStream>,
    pool: &'a Pool<StreamPair>,
}

impl Drop for SessionStreams<'_> {
    fn drop(&mut self) {
        // Also covers early build errors, before a SelfFeedingSession exists.
        // Try both streams even if one fails. Never recycle uncertain work.
        let transfer = self.transfer.synchronize();
        let compute = self.compute.synchronize();
        let pair = (Arc::clone(&self.compute), Arc::clone(&self.transfer));
        if transfer.is_ok() && compute.is_ok() {
            self.pool.put(pair);
        } else {
            eprintln!(
                "cuda stream pool: drain failed; leaking stream pair: \
                 transfer={transfer:?}, compute={compute:?}"
            );
            // An uncertain stream must stay alive and outside the pool.
            #[allow(clippy::mem_forget)]
            std::mem::forget(pair);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn checkout_is_exclusive_and_return_reuses_the_item() {
        let pool = Pool::new(vec![11, 22]);
        let first = pool.take();
        let second = pool.take();
        assert_ne!(first, second);
        pool.put(first);
        assert_eq!(pool.take(), first);
        pool.put(second);
        assert_eq!(pool.take(), second);
    }

    #[test]
    fn empty_pool_blocks_and_return_wakes_the_waiter() {
        let pool = Arc::new(Pool::new(vec![17]));
        let item = pool.take();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (taken_tx, taken_rx) = mpsc::channel();
        let waiter_pool = Arc::clone(&pool);
        let waiter = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            taken_tx.send(waiter_pool.take()).unwrap();
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let blocked = taken_rx.recv_timeout(Duration::from_millis(50));
        pool.put(item);
        let returned = taken_rx.recv_timeout(Duration::from_secs(5));
        assert_eq!(blocked, Err(mpsc::RecvTimeoutError::Timeout));
        assert_eq!(returned.unwrap(), 17);
        waiter.join().unwrap();
    }
}
