"""Fixed FP16 math, explicitly strided Triton primitives for RQ2 graphs."""
import triton
import triton.language as tl


@triton.jit
def rms_kernel(X, W, O, RES, R: tl.constexpr, C: tl.constexpr,
               S0: tl.constexpr, S1: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
               ADD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X+r*S0+c*S1, c<C, 0).to(tl.float32)
    if ADD:
        x += tl.load(RES+r*C+c, c<C, 0).to(tl.float32)
        tl.store(RES+r*C+c, x, c<C)
    w = tl.load(W+c, c<C, 0).to(tl.float32)
    y = x*tl.rsqrt(tl.sum(x*x, 0)/C+EPS)*w
    tl.store(O+r*C+c, y, c<C)


@triton.jit
def silu_kernel(X, O, R: tl.constexpr, C: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    r, c = i//C, i%C
    a = tl.load(X+r*S0+c*S1, i<R*C, 0).to(tl.float32)
    b = tl.load(X+r*S0+(c+C)*S1, i<R*C, 0).to(tl.float32)
    tl.store(O+i, a/(1+tl.exp(-a))*b, i<R*C)


@triton.jit
def attention_kernel(Q, K, V, O, LENS, QLEN: tl.constexpr, KLEN: tl.constexpr,
                     HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                     KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
                     SCALE: tl.constexpr, BM: tl.constexpr = 16, BN: tl.constexpr = 64):
    # Keys/values logical [B,KV,Hkv,D]; both NHD and HND are accepted directly.
    bhead = tl.program_id(1)
    b, h = bhead//HQ, bhead%HQ
    valid_kv=tl.load(LENS+b).to(tl.int32)
    kh = h//(HQ//HK)
    m = tl.program_id(0)*BM+tl.arange(0,BM)
    n = tl.arange(0,BN)
    d = tl.arange(0,D)
    q = tl.load(Q+((b*QLEN+m[:,None])*HQ+h)*D+d[None,:], m[:,None]<QLEN, 0)
    max_prev = tl.full((BM,), -float('inf'), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM,D), 0, tl.float32)
    for start in range(tl.cdiv(valid_kv,BN)):
        nn = start*BN+n
        k = tl.load(K+b*KB+nn[None,:]*KT+kh*KH+d[:,None], nn[None,:]<valid_kv, 0)
        scores = tl.dot(q,k)*SCALE
        legal = (nn[None,:]<valid_kv) & (nn[None,:] <= valid_kv-QLEN+m[:,None])
        scores = tl.where(legal, scores, -float('inf'))
        maximum = tl.maximum(max_prev, tl.max(scores,1))
        p = tl.exp(scores-maximum[:,None])
        alpha = tl.exp(max_prev-maximum)
        denom = denom*alpha+tl.sum(p,1)
        v = tl.load(V+b*KB+nn[:,None]*KT+kh*KH+d[None,:], nn[:,None]<valid_kv, 0)
        acc = acc*alpha[:,None]+tl.dot(p.to(tl.float16),v)
        max_prev = maximum
    out = acc/denom[:,None]
    tl.store(O+((b*QLEN+m[:,None])*HQ+h)*D+d[None,:], out, m[:,None]<QLEN)


def rms(x, w, out, eps, residual=None):
    return rms_kernel[(x.shape[0],)](x,w,out,residual if residual is not None else out,
                                    *x.shape,*x.stride(),eps,
                                    triton.next_power_of_2(x.shape[1]),residual is not None)


def silu(x, out):
    return silu_kernel[(triton.cdiv(out.numel(),256),)](x,out,*out.shape,*x.stride())


def attention(q, k, v, out, lengths):
    b,t,h,d=q.shape
    return attention_kernel[(triton.cdiv(t,16),b*h)](
        q,k,v,out,lengths,t,k.shape[1],h,k.shape[2],d,*k.stride()[:3],d**-.5,num_warps=4)


@triton.jit
def append_kernel(X, CACHE, LENS, Q: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  B: tl.constexpr, CS0: tl.constexpr, CS1: tl.constexpr,
                  CS2: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
                  XS2: tl.constexpr, BLOCK: tl.constexpr = 256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    request=i//(Q*H*D);token=(i//(H*D))%Q;head=(i//D)%H;d=i%D
    length=tl.load(LENS+request,request<B,0)
    value=tl.load(X+(request*Q+token)*XS0+head*XS1+d*XS2,i<B*Q*H*D,0)
    tl.store(CACHE+request*CS0+(length-Q+token)*CS1+head*CS2+d,value,i<B*Q*H*D)


def append(x, cache, lengths, query_length):
    b,kv,h,d=cache.shape
    return append_kernel[(triton.cdiv(b*query_length*h*d,256),)](
        x,cache,lengths,query_length,h,d,b,*cache.stride()[:3],*x.stride())
