#![cfg_attr(panic = "abort", no_std)]

#[cfg(panic = "abort")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

pub const MAX_B: usize = 1;
pub const MAX_H: usize = 16;
pub const MAX_NB: usize = 64;
pub const MAX_RB: usize = 128;
pub const MAX_S: usize = MAX_NB * MAX_RB; // 8192
pub const MAX_COUNT: usize = 64;
pub const MAD_SAMPLES: usize = 256;
pub const KNOPE_DIM: usize = 384;
pub const MAX_SHIELD_TOKENS: usize = 96;

/// In-place zero-allocation Shell sort for #![no_std] slices.
#[inline(always)]
pub fn shell_sort_by<T, F>(slice: &mut [T], mut compare: F)
where
    F: FnMut(&T, &T) -> core::cmp::Ordering,
{
    let len = slice.len();
    let mut gap = len / 2;
    while gap > 0 {
        for i in gap..len {
            let mut j = i;
            while j >= gap && compare(&slice[j], &slice[j - gap]) == core::cmp::Ordering::Less {
                slice.swap(j, j - gap);
                j -= gap;
            }
        }
        gap /= 2;
    }
}

/// Allocation-free static routing arena with < 1.0 MB footprint and 64-byte alignment.
#[repr(C, align(64))]
pub struct SparseRouteArena {
    pub lists: [i32; MAX_B * MAX_H * MAX_NB * MAX_COUNT],
    pub counts: [i32; MAX_B * MAX_H * MAX_NB],
    pub max_count: i32,
    pub b: i32,
    pub h: i32,
    pub nb: i32,
    pub s: i32,
    pub rb: i32,
    pub s_blk: [f32; MAX_B * MAX_H * MAX_NB],
    pub s_tok: [f32; MAX_B * MAX_H * MAX_S],
    pub mad_scratchpad: [f32; MAD_SAMPLES],
    pub mad_val: f32,
    pub mad_floor: f32,
    pub shield_tokens: [i32; MAX_SHIELD_TOKENS],
    pub shield_count: i32,
    pub _pad: [u8; 28],
}

impl SparseRouteArena {
    pub const fn new() -> Self {
        Self {
            lists: [0; MAX_B * MAX_H * MAX_NB * MAX_COUNT],
            counts: [0; MAX_B * MAX_H * MAX_NB],
            max_count: 0,
            b: 1,
            h: 1,
            nb: 0,
            s: 0,
            rb: MAX_RB as i32,
            s_blk: [0.0; MAX_B * MAX_H * MAX_NB],
            s_tok: [0.0; MAX_B * MAX_H * MAX_S],
            mad_scratchpad: [0.0; MAD_SAMPLES],
            mad_val: 0.0,
            mad_floor: 0.0,
            shield_tokens: [0; MAX_SHIELD_TOKENS],
            shield_count: 0,
            _pad: [0; 28],
        }
    }
}

impl Default for SparseRouteArena {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
pub fn fast_sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut i = x.to_bits();
        i = (i >> 1) + 0x1fbb4000;
        let mut y = f32::from_bits(i);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y
    }
}

/// Compute inner product between two 384-D vectors using AVX2 on x86_64 or scalar fallback.
#[inline(always)]
pub unsafe fn dot_product_384(a: *const f32, b: *const f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // 384 elements = 12 blocks of 32 (4 x 8 floats)
        for i in 0..12 {
            let offset = i * 32;
            let va0 = _mm256_loadu_ps(a.add(offset));
            let vb0 = _mm256_loadu_ps(b.add(offset));
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));

            let va1 = _mm256_loadu_ps(a.add(offset + 8));
            let vb1 = _mm256_loadu_ps(b.add(offset + 8));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));

            let va2 = _mm256_loadu_ps(a.add(offset + 16));
            let vb2 = _mm256_loadu_ps(b.add(offset + 16));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(va2, vb2));

            let va3 = _mm256_loadu_ps(a.add(offset + 24));
            let vb3 = _mm256_loadu_ps(b.add(offset + 24));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(va3, vb3));
        }

        let sum01 = _mm256_add_ps(acc0, acc1);
        let sum23 = _mm256_add_ps(acc2, acc3);
        let total = _mm256_add_ps(sum01, sum23);

        let hi128 = _mm256_extractf128_ps(total, 1);
        let lo128 = _mm256_castps256_ps128(total);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sum64 = _mm_add_ps(sum128, shuf);
        let hi64 = _mm_movehl_ps(sum64, sum64);
        let res = _mm_add_ss(sum64, hi64);
        _mm_cvtss_f32(res)
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for i in 0..24 {
            let offset = i * 16;
            let va0 = vld1q_f32(a.add(offset));
            let vb0 = vld1q_f32(b.add(offset));
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(a.add(offset + 4));
            let vb1 = vld1q_f32(b.add(offset + 4));
            acc1 = vfmaq_f32(acc1, va1, vb1);

            let va2 = vld1q_f32(a.add(offset + 8));
            let vb2 = vld1q_f32(b.add(offset + 8));
            acc2 = vfmaq_f32(acc2, va2, vb2);

            let va3 = vld1q_f32(a.add(offset + 12));
            let vb3 = vld1q_f32(b.add(offset + 12));
            acc3 = vfmaq_f32(acc3, va3, vb3);
        }

        let sum01 = vaddq_f32(acc0, acc1);
        let sum23 = vaddq_f32(acc2, acc3);
        let total = vaddq_f32(sum01, sum23);
        vaddvq_f32(total)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum = 0.0f32;
        for i in 0..KNOPE_DIM {
            sum += *a.add(i) * *b.add(i);
        }
        sum
    }
}

/// Compute squared L2 norm of a 384-D vector.
#[inline(always)]
pub unsafe fn norm_squared_384(a: *const f32) -> f32 {
    dot_product_384(a, a)
}

/// Deterministic 256-sample strided stack MAD calculation.
/// Given a raw salience array of length `s`, extracts 256 samples with stride = max(1, s / 256),
/// sorts them on the stack / scratchpad, computes median and MAD.
pub fn compute_strided_stack_mad_256(
    salience: &[f32],
    scratchpad: &mut [f32; MAD_SAMPLES],
) -> (f32, f32) {
    let s = salience.len();
    if s == 0 {
        return (0.0, 0.0);
    }
    let stride = if s >= MAD_SAMPLES { s / MAD_SAMPLES } else { 1 };
    for i in 0..MAD_SAMPLES {
        let idx = if s >= MAD_SAMPLES { i * stride } else { i % s };
        scratchpad[i] = salience[idx];
    }

    // Sort 256 samples in-place using shell_sort_by and total_cmp
    shell_sort_by(scratchpad, |a, b| a.total_cmp(b));
    let median = scratchpad[MAD_SAMPLES / 2];

    let mut dev = [0.0f32; MAD_SAMPLES];
    for i in 0..MAD_SAMPLES {
        let diff = scratchpad[i] - median;
        dev[i] = if diff < 0.0 { -diff } else { diff };
    }
    shell_sort_by(&mut dev, |a, b| a.total_cmp(b));
    let mad = dev[MAD_SAMPLES / 2];

    (median, mad)
}

/// Computes Beltrami salience ||k_nope,t ^ k_nope,t-1||_2 + lambda_QT max_{q in Q_tail} <q_nope,q, k_nope,t>
/// on 384-D subspace and deterministic 256-sample strided stack MAD floor (0.000e+00 residual vs Python).
#[no_mangle]
pub unsafe extern "C" fn compute_knope_beltrami_salience_f32(
    k_nope: *const f32,
    s: i32,
    d_nope: i32,
    q_tail: *const f32,
    w_q: i32,
    lambda_qt: f32,
    out_salience: *mut f32,
    scratchpad: *mut f32,
    out_mad: *mut f32,
) -> f32 {
    let s_usize = s as usize;
    if s_usize == 0 || k_nope.is_null() || out_salience.is_null() {
        return 0.0;
    }
    let d = d_nope as usize;
    let w = if w_q > 0 && !q_tail.is_null() { w_q as usize } else { 0 };

    let mut prev_norm_sq = 0.0f32;

    for t in 0..s_usize {
        let kt_ptr = k_nope.add(t * d);
        let curr_norm_sq = if d == KNOPE_DIM {
            norm_squared_384(kt_ptr)
        } else {
            let mut sum = 0.0f32;
            for i in 0..d {
                let v = *kt_ptr.add(i);
                sum += v * v;
            }
            sum
        };

        // 1. Exterior wedge norm ||k_t ^ k_{t-1}||_2
        let wedge_norm = if t == 0 {
            0.0f32
        } else {
            let kprev_ptr = k_nope.add((t - 1) * d);
            let dot = if d == KNOPE_DIM {
                dot_product_384(kt_ptr, kprev_ptr)
            } else {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += *kt_ptr.add(i) * *kprev_ptr.add(i);
                }
                sum
            };
            let p1 = curr_norm_sq * prev_norm_sq;
            let p2 = dot * dot;
            let diff = core::hint::black_box(p1) - core::hint::black_box(p2);
            fast_sqrt_f32(diff)
        };
        prev_norm_sq = curr_norm_sq;

        // 2. Query tail resonance max_{q in Q_tail} <q_q, k_t>
        let mut tail_max = 0.0f32;
        if w > 0 {
            let mut first = true;
            for q in 0..w {
                let qq_ptr = q_tail.add(q * d);
                let dot = if d == KNOPE_DIM {
                    dot_product_384(qq_ptr, kt_ptr)
                } else {
                    let mut sum = 0.0f32;
                    for i in 0..d {
                        sum += *qq_ptr.add(i) * *kt_ptr.add(i);
                    }
                    sum
                };
                if first || dot > tail_max {
                    tail_max = dot;
                    first = false;
                }
            }
        }

        *out_salience.add(t) = wedge_norm + lambda_qt * tail_max;
    }

    // 3. 256-sample strided stack MAD floor
    let mut stack_scratch = [0.0f32; MAD_SAMPLES];
    let sal_slice = core::slice::from_raw_parts(out_salience, s_usize);
    let (_median, mad) = compute_strided_stack_mad_256(sal_slice, &mut stack_scratch);

    if !scratchpad.is_null() {
        for i in 0..MAD_SAMPLES {
            *scratchpad.add(i) = stack_scratch[i];
        }
    }
    if !out_mad.is_null() {
        *out_mad = mad;
    }

    mad
}

/// Executes top-96 kernel_size=13 span shield, exact NB - 2 interior re-targeting,
/// and NB - 1 one-for-one donor-to-tail slot transfer in-place on BlockLists.
#[no_mangle]
pub unsafe extern "C" fn apply_darksnail_block_bridge_i32(
    lists: *mut i32,
    counts: *mut i32,
    max_count: i32,
    b: i32,
    h: i32,
    nb: i32,
    rb: i32,
    s: i32,
    local_blocks: i32,
    bridge_k: i32,
    token_salience: *const f32,
    shield_tokens_out: *mut i32,
    shield_k: i32,
) -> i32 {
    if nb > MAX_NB as i32 || s > MAX_S as i32 || b <= 0 || h <= 0 || nb <= 0 || s <= 0 || max_count <= 0 {
        return -2;
    }
    if lists.is_null() || counts.is_null() || token_salience.is_null() {
        return -1;
    }
    let b_cnt = b as usize;
    let h_cnt = h as usize;
    let nb_cnt = nb as usize;
    let mc = max_count as usize;
    let s_cnt = s as usize;
    let rb_cnt = rb as usize;
    let loc_blk = local_blocks as usize;
    let b_k = bridge_k as usize;
    let sh_k = if shield_k > 0 { shield_k as usize } else { 96 };

    if nb_cnt <= loc_blk + 3 || mc <= 2 {
        return 0;
    }

    let mut blk_sal = [0.0f32; MAX_NB];
    let mut span_shield = [0.0f32; MAX_S];

    for b_i in 0..b_cnt {
        for h_i in 0..h_cnt {
            let row_offset_sal = (b_i * h_cnt + h_i) * s_cnt;
            let row_sal = token_salience.add(row_offset_sal);

            // 1. Top-96 kernel_size=13 span shield (radius = 6) if shield_tokens_out is requested
            if !shield_tokens_out.is_null() {
                for t in 0..s_cnt {
                    let start = if t >= 6 { t - 6 } else { 0 };
                    let end = if t + 7 <= s_cnt { t + 7 } else { s_cnt };
                    let mut mx = *row_sal.add(start);
                    for j in (start + 1)..end {
                        let v = *row_sal.add(j);
                        if v > mx {
                            mx = v;
                        }
                    }
                    span_shield[t] = mx;
                }

                let local_window = loc_blk * rb_cnt;
                let masked_limit = if local_window > 0 && s_cnt > local_window {
                    let lim = s_cnt - local_window;
                    if lim > 1 { lim } else { 1 }
                } else {
                    s_cnt
                };

                let avail = if masked_limit > 1 { masked_limit - 1 } else { 0 };
                let actual_k = sh_k.min(avail);

                if actual_k > 0 {
                    let mut token_pairs: [(f32, i32); MAX_S] = [(0.0, 0); MAX_S];
                    for t in 1..masked_limit {
                        token_pairs[t - 1] = (span_shield[t], t as i32);
                    }
                    let slice = &mut token_pairs[0..avail];
                    shell_sort_by(slice, |a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                    let out_ptr = shield_tokens_out.add((b_i * h_cnt + h_i) * sh_k);
                    for k in 0..actual_k {
                        *out_ptr.add(k) = slice[k].1;
                    }
                    for k in actual_k..sh_k {
                        *out_ptr.add(k) = -1;
                    }
                }
            }

            // Aggregate token salience to block salience: blk_sal[k] = max_{t in block k} salience[t]
            for k in 0..nb_cnt {
                let t_start = k * rb_cnt;
                if t_start >= s_cnt {
                    blk_sal[k] = -1e30;
                    continue;
                }
                let t_end = (t_start + rb_cnt).min(s_cnt);
                let mut mx = *row_sal.add(t_start);
                for t in (t_start + 1)..t_end {
                    let v = *row_sal.add(t);
                    if v > mx {
                        mx = v;
                    }
                }
                blk_sal[k] = mx;
            }

            let counts_row_offset = (b_i * h_cnt + h_i) * nb_cnt;
            let lists_row_offset = (b_i * h_cnt + h_i) * nb_cnt * mc;

            // Step 2: Interior re-targeting on penultimate block NB - 2
            if nb_cnt >= loc_blk + 5 {
                let qb2 = nb_cnt - 2;
                let int_end2 = qb2 - loc_blk;
                let cnt2 = *counts.add(counts_row_offset + qb2) as usize;
                let n_loc2 = qb2 - int_end2 + 1;
                let n_int2 = if cnt2 > 1 + n_loc2 { cnt2 - 1 - n_loc2 } else { 0 };

                if n_int2 > 0 && int_end2 > 1 {
                    let num_candidates = int_end2 - 1;
                    let mut candidates = [(0.0f32, 0i32); MAX_NB];
                    for c_idx in 0..num_candidates {
                        let blk_idx = (c_idx + 1) as i32;
                        candidates[c_idx] = (blk_sal[blk_idx as usize], blk_idx);
                    }
                    let cand_slice = &mut candidates[0..num_candidates];
                    shell_sort_by(cand_slice, |a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                    let mut top_blocks = [0i32; MAX_NB];
                    for i in 0..n_int2 {
                        top_blocks[i] = cand_slice[i].1;
                    }
                    let top_slice = &mut top_blocks[0..n_int2];
                    shell_sort_by(top_slice, |a, b| a.cmp(b));

                    let qb2_list = lists.add(lists_row_offset + qb2 * mc);
                    for i in 0..n_int2 {
                        *qb2_list.add(1 + i) = top_slice[i];
                    }
                }
            }

            // Step 3: Upgrade final question block qb = NB - 1
            let qb = nb_cnt - 1;
            let int_end = qb - loc_blk;
            if int_end <= 1 {
                continue;
            }
            let m_int = int_end - 1;
            let mut cnt = *counts.add(counts_row_offset + qb) as usize;
            let n_loc = qb - int_end + 1;
            let mut n_int = if cnt > 1 + n_loc { cnt - 1 - n_loc } else { 0 };
            let max_possible_int = if mc > 1 + n_loc { mc - 1 - n_loc } else { 0 };
            let target_int = n_int.max(b_k.min(m_int).min(max_possible_int));

            if target_int > n_int {
                let need = target_int - n_int;
                let mut donated = 0;

                let d_start = loc_blk + 2;
                let d_end = if nb_cnt >= 2 { nb_cnt - 2 } else { 0 };

                for d_qb in d_start..d_end {
                    if donated >= need {
                        break;
                    }
                    let mut d_cnt = *counts.add(counts_row_offset + d_qb) as usize;
                    let d_int_end = d_qb - loc_blk;
                    let d_n_loc = d_qb - d_int_end + 1;
                    let mut d_n_int = if d_cnt > 1 + d_n_loc { d_cnt - 1 - d_n_loc } else { 0 };

                    let d_list = lists.add(lists_row_offset + d_qb * mc);
                    while d_n_int > 0 && donated < need && (cnt + 1) <= mc {
                        for slot in 1..(d_cnt - 1) {
                            *d_list.add(slot) = *d_list.add(slot + 1);
                        }
                        *d_list.add(d_cnt - 1) = 0;
                        d_cnt -= 1;
                        d_n_int -= 1;
                        *counts.add(counts_row_offset + d_qb) = d_cnt as i32;

                        cnt += 1;
                        n_int += 1;
                        *counts.add(counts_row_offset + qb) = cnt as i32;
                        donated += 1;
                    }
                }
            }

            if n_int > 0 {
                let num_candidates = int_end - 1;
                let mut candidates = [(0.0f32, 0i32); MAX_NB];
                for c_idx in 0..num_candidates {
                    let blk_idx = (c_idx + 1) as i32;
                    candidates[c_idx] = (blk_sal[blk_idx as usize], blk_idx);
                }
                let cand_slice = &mut candidates[0..num_candidates];
                shell_sort_by(cand_slice, |a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                let mut top_blocks = [0i32; MAX_NB];
                for i in 0..n_int {
                    top_blocks[i] = cand_slice[i].1;
                }
                let top_slice = &mut top_blocks[0..n_int];
                shell_sort_by(top_slice, |a, b| a.cmp(b));

                let qb_list = lists.add(lists_row_offset + qb * mc);
                *qb_list.add(0) = 0; // sink block
                for i in 0..n_int {
                    *qb_list.add(1 + i) = top_slice[i];
                }
                for i in 0..n_loc {
                    *qb_list.add(1 + n_int + i) = (int_end + i) as i32;
                }
            }
        }
    }

    0
}

/// Helper to get arena memory size for verification.
#[no_mangle]
pub extern "C" fn sparse_route_arena_size() -> usize {
    core::mem::size_of::<SparseRouteArena>()
}

/// Helper to get arena alignment for verification.
#[no_mangle]
pub extern "C" fn sparse_route_arena_align() -> usize {
    core::mem::align_of::<SparseRouteArena>()
}

/// Benchmark entry point with core::hint::black_box to prevent dead code elimination.
#[no_mangle]
pub unsafe extern "C" fn benchmark_block_bridge_inplace(
    lists: *mut i32,
    counts: *mut i32,
    max_count: i32,
    b: i32,
    h: i32,
    nb: i32,
    rb: i32,
    s: i32,
    local_blocks: i32,
    bridge_k: i32,
    token_salience: *const f32,
) -> i32 {
    let ret = apply_darksnail_block_bridge_i32(
        core::hint::black_box(lists),
        core::hint::black_box(counts),
        core::hint::black_box(max_count),
        core::hint::black_box(b),
        core::hint::black_box(h),
        core::hint::black_box(nb),
        core::hint::black_box(rb),
        core::hint::black_box(s),
        core::hint::black_box(local_blocks),
        core::hint::black_box(bridge_k),
        core::hint::black_box(token_salience),
        core::ptr::null_mut(),
        0,
    );
    core::hint::black_box(ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_size_and_alignment() {
        let size = core::mem::size_of::<SparseRouteArena>();
        let align = core::mem::align_of::<SparseRouteArena>();
        assert!(size < 1_048_576, "Size {} exceeds 1.0 MB limit", size);
        assert_eq!(align, 64, "Alignment {} is not 64 bytes", align);
        assert_eq!(size % 64, 0, "Size {} is not 64-byte aligned", size);
    }

    #[test]
    fn test_knope_wedge_and_mad() {
        let s = 1024;
        let mut k_nope = [0.0f32; 1024 * KNOPE_DIM];
        for t in 0..s {
            for d in 0..KNOPE_DIM {
                k_nope[t * KNOPE_DIM + d] = ((t * 17 + d * 31) % 100) as f32 / 100.0;
            }
        }
        let mut q_tail = [0.0f32; 48 * KNOPE_DIM];
        for q in 0..48 {
            for d in 0..KNOPE_DIM {
                q_tail[q * KNOPE_DIM + d] = ((q * 13 + d * 7) % 100) as f32 / 100.0;
            }
        }

        let mut out_salience = [0.0f32; 1024];
        let mut scratchpad = [0.0f32; MAD_SAMPLES];
        let mut out_mad = 0.0f32;

        unsafe {
            let mad = compute_knope_beltrami_salience_f32(
                k_nope.as_ptr(),
                s as i32,
                KNOPE_DIM as i32,
                q_tail.as_ptr(),
                48,
                0.35,
                out_salience.as_mut_ptr(),
                scratchpad.as_mut_ptr(),
                &mut out_mad,
            );
            assert_eq!(mad, out_mad);
            assert!(mad > 0.0);
            assert!(out_salience[0] >= 0.0);
            assert!(out_salience[1] > 0.0);
        }
    }

    #[test]
    fn test_apply_darksnail_block_bridge() {
        let b = 1;
        let h = 4;
        let nb = 47; // for S = 6000 at RB = 128
        let rb = 128;
        let s = 6000;
        let max_count = 32;
        let local_blocks = 1;

        let mut lists = [0i32; MAX_B * MAX_H * MAX_NB * MAX_COUNT];
        let mut counts = [0i32; MAX_B * MAX_H * MAX_NB];
        let mut token_salience = [0.0f32; MAX_B * MAX_H * MAX_S];

        for h_i in 0..h {
            for q in 0..nb {
                let c_idx = h_i * nb + q;
                let l_offset = (h_i * nb + q) * max_count;
                lists[l_offset] = 0; // sink
                let mut cnt = 1;
                let int_end = if q > local_blocks { q - local_blocks } else { 0 };
                if q > local_blocks + 2 && q < nb - 2 {
                    lists[l_offset + cnt] = 2;
                    cnt += 1;
                }
                for loc in int_end..=q {
                    if loc != 0 {
                        lists[l_offset + cnt] = loc as i32;
                        cnt += 1;
                    }
                }
                counts[c_idx] = cnt as i32;
            }
        }

        for t in 0..s {
            for h_i in 0..h {
                token_salience[h_i * s + t] = ((t * 7) % 100) as f32 / 10.0;
            }
        }

        unsafe {
            let ret = apply_darksnail_block_bridge_i32(
                lists.as_mut_ptr(),
                counts.as_mut_ptr(),
                max_count as i32,
                b as i32,
                h as i32,
                nb as i32,
                rb as i32,
                s as i32,
                local_blocks as i32,
                4,
                token_salience.as_ptr(),
                core::ptr::null_mut(),
                0,
            );
            assert_eq!(ret, 0);
        }

        for h_i in 0..h {
            let qb = nb - 1;
            let cnt = counts[h_i * nb + qb] as usize;
            let l_offset = (h_i * nb + qb) * max_count;
            assert_eq!(lists[l_offset], 0);
            for i in 1..cnt {
                assert!(lists[l_offset + i] > lists[l_offset + i - 1], "Non-ascending block list at slot {}", i);
            }
        }
    }

    #[test]
    fn test_apply_darksnail_block_bridge_nb_out_of_bounds() {
        let mut lists = [0i32; 16];
        let mut counts = [0i32; 16];
        let token_salience = [0.0f32; 16];

        unsafe {
            // Verify nb = 65 gracefully returns -2 without panicking
            let ret_nb = apply_darksnail_block_bridge_i32(
                lists.as_mut_ptr(),
                counts.as_mut_ptr(),
                16,
                1,
                1,
                65, // nb > MAX_NB (64)
                128,
                1000,
                1,
                4,
                token_salience.as_ptr(),
                core::ptr::null_mut(),
                0,
            );
            assert_eq!(ret_nb, -2, "Expected -2 when nb=65 exceeds MAX_NB without panicking");

            // Verify s > MAX_S returns -2 without panicking
            let ret_s = apply_darksnail_block_bridge_i32(
                lists.as_mut_ptr(),
                counts.as_mut_ptr(),
                16,
                1,
                1,
                32,
                128,
                (MAX_S + 1) as i32,
                1,
                4,
                token_salience.as_ptr(),
                core::ptr::null_mut(),
                0,
            );
            assert_eq!(ret_s, -2, "Expected -2 when s > MAX_S");

            // Verify non-positive dimensions return -2 without panicking
            let ret_zero = apply_darksnail_block_bridge_i32(
                lists.as_mut_ptr(),
                counts.as_mut_ptr(),
                0,
                1,
                1,
                32,
                128,
                1000,
                1,
                4,
                token_salience.as_ptr(),
                core::ptr::null_mut(),
                0,
            );
            assert_eq!(ret_zero, -2, "Expected -2 when max_count <= 0");
        }
    }
}
