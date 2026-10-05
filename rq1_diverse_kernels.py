"""Fixed-arithmetic Triton producers with explicit physical output mappings."""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, W, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                AS0: tl.constexpr, AS1: tl.constexpr, AS2: tl.constexpr,
                WS0: tl.constexpr, WS1: tl.constexpr, WS2: tl.constexpr,
                OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr,
                SEG1: tl.constexpr, SEG2: tl.constexpr, SEGMENTED: tl.constexpr,
                GROUPS_PER_REQUEST: tl.constexpr, REQUEST_STRIDE: tl.constexpr,
                BM: tl.constexpr = 32, BN: tl.constexpr = 64, BK: tl.constexpr = 32):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    group = tl.program_id(2)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + group*AS0 + m[:, None]*AS1 + kk[None, :]*AS2,
                    (m[:, None] < M) & (kk[None, :] < K), 0)
        w = tl.load(W + group*WS0 + kk[:, None]*WS1 + n[None, :]*WS2,
                    (kk[:, None] < K) & (n[None, :] < N), 0)
        acc += tl.dot(a, w)
    if SEGMENTED:
        # Direct projection into three separately dense component buffers in
        # one backing allocation, using the same GEMM tile/launch as packed-row.
        width = tl.where(n < SEG1, SEG1,
                         tl.where(n < SEG1 + SEG2, SEG2, N - SEG1 - SEG2))
        prefix = tl.where(n < SEG1, 0, tl.where(n < SEG1 + SEG2, SEG1, SEG1 + SEG2))
        offset = group*M*N + prefix[None, :]*M + m[:, None]*width[None, :] + (n-prefix)[None, :]
    else:
        if GROUPS_PER_REQUEST:
            base=(group//GROUPS_PER_REQUEST)*REQUEST_STRIDE+(group%GROUPS_PER_REQUEST)*OS0
        else:
            base=group*OS0
        offset = base + m[:, None]*OS1 + n[None, :]*OS2
    tl.store(O + offset, acc.to(O.dtype.element_ty), (m[:, None] < M) & (n[None, :] < N))


def gemm(a, w, out, segments=()):
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if w.ndim == 2:
        w = w.unsqueeze(0)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    groups, m, k = a.shape
    n = w.shape[-1]
    out_strides=out.stride()[1:] if out.ndim==4 else out.stride()
    gemm_kernel[(triton.cdiv(m,32), triton.cdiv(n,64), groups)](
        a, w, out, m, n, k, *a.stride(), *w.stride(), *out_strides,
        segments[0] if segments else 0, segments[1] if segments else 0,
        bool(segments),out.shape[1] if out.ndim==4 else 0,
        out.stride(0) if out.ndim==4 else 0,num_warps=4)


@triton.jit
def gather_kernel(X, IDS, O, ROWS: tl.constexpr, COLS: tl.constexpr,
                  XS: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
                  BLOCK: tl.constexpr = 256):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    row, col = i // COLS, i % COLS
    source = tl.load(IDS + row, row < ROWS, 0)
    value = tl.load(X + source*XS + col, i < ROWS*COLS, 0)
    tl.store(O + row*OS0 + col*OS1, value, i < ROWS*COLS)


def gather(x, indices, out):
    rows, cols = out.shape
    gather_kernel[(triton.cdiv(rows*cols,256),)](
        x, indices, out, rows, cols, x.stride(0), *out.stride())


@triton.jit
def reduce_kernel(X, O, ROWS: tl.constexpr, COLS: tl.constexpr,
                  OS0: tl.constexpr, OS1: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0,BLOCK)
    x = tl.load(X + row*COLS + cols, cols < COLS, 0).to(tl.float32)
    square = tl.sum(x*x,0)/COLS
    y = x*tl.rsqrt(square+1e-6)
    tl.store(O + row*OS0 + cols*OS1, y.to(O.dtype.element_ty), cols < COLS)


def rms_emit(x, out):
    reduce_kernel[(x.shape[0],)](x, out, *x.shape, *out.stride(),
                              triton.next_power_of_2(x.shape[1]))


@triton.jit
def paged_append_kernel(X, O, PAGES, B: tl.constexpr, KV: tl.constexpr,
                        WRITE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                        PAGE: tl.constexpr, NP: tl.constexpr, HND: tl.constexpr,
                        BLOCK: tl.constexpr = 256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=i%D;head=(i//D)%H;token=(i//(H*D))%WRITE+KV-WRITE
    request=i//(WRITE*H*D)
    page=tl.load(PAGES+request*NP+token//PAGE,request<B,0)
    if HND:
        offset=((page*H+head)*PAGE+token%PAGE)*D+d
    else:
        offset=((page*PAGE+token%PAGE)*H+head)*D+d
    value=tl.load(X+((request*KV+token)*H+head)*D+d,i<B*WRITE*H*D,0)
    tl.store(O+offset,value,i<B*WRITE*H*D)


def paged_append(x,out,pages,write,page,hnd):
    b,kv,h,d=x.shape
    paged_append_kernel[(triton.cdiv(b*write*h*d,256),)](
        x,out,pages,b,kv,write,h,d,page,pages.numel()//b,hnd)
