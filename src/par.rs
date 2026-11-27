//! Minimal fork-join helpers. With the `std` feature they use scoped threads;
//! without it everything runs sequentially.

/// Transforms smaller than this always run on one thread.
pub(crate) const PAR_MIN_LEN: usize = 1 << 18;

/// Number of worker threads to use for a job of `len` elements, given a
/// user setting (`0` = automatic).
pub(crate) fn threads_for(len: usize, setting: usize) -> usize {
    if setting != 0 {
        return setting;
    }
    if len < PAR_MIN_LEN {
        return 1;
    }
    available().min(len / (PAR_MIN_LEN / 4)).max(1)
}

#[cfg(feature = "std")]
fn available() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

#[cfg(not(feature = "std"))]
fn available() -> usize {
    1
}

/// Run `a` and `b`, in parallel when `threads > 1`.
#[cfg(feature = "std")]
pub(crate) fn join<A: FnOnce() + Send, B: FnOnce() + Send>(threads: usize, a: A, b: B) {
    if threads > 1 {
        std::thread::scope(|s| {
            s.spawn(a);
            b();
        });
    } else {
        a();
        b();
    }
}

#[cfg(not(feature = "std"))]
pub(crate) fn join<A: FnOnce() + Send, B: FnOnce() + Send>(_threads: usize, a: A, b: B) {
    a();
    b();
}

/// Apply `f` to every chunk, spreading the chunks over up to `threads` threads.
#[cfg(feature = "std")]
pub(crate) fn for_each_chunk<T: Send, F: Fn(&mut [T]) + Sync>(data: &mut [T], chunk: usize, threads: usize, f: F) {
    if threads <= 1 || data.len() <= chunk {
        data.chunks_mut(chunk).for_each(f);
        return;
    }
    let chunks = data.len().div_ceil(chunk);
    let per = chunks.div_ceil(threads) * chunk;
    std::thread::scope(|s| {
        for part in data.chunks_mut(per) {
            let f = &f;
            s.spawn(move || part.chunks_mut(chunk).for_each(f));
        }
    });
}

#[cfg(not(feature = "std"))]
pub(crate) fn for_each_chunk<T: Send, F: Fn(&mut [T]) + Sync>(data: &mut [T], chunk: usize, _threads: usize, f: F) {
    data.chunks_mut(chunk).for_each(f);
}
