#include "C/surrogate.in"

template<typename scalar_t> __global__ void Leaky_integrate(
    const scalar_t * x, 
    scalar_t * spike,
    const scalar_t tau, 
    const scalar_t v_th,
    const scalar_t v_reset,
    const int batch,
    const int step,
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = (idx_th - idx_th % dim) * step + (idx_th % dim);
    scalar_t potential = 0.0;

    for (int i = 0;i < step;++i, posi += dim) {
        potential = x[posi] + potential / tau;
        spike[posi] = potential >= v_th ? 1 : 0;
        potential = potential >= v_th ? v_reset : potential;
    }
}

template<typename scalar_t> __global__ void Leaky_integrate_FP(
    const scalar_t * x, 
    scalar_t * psps, 
    scalar_t * spike, 
    const scalar_t tau, 
    const scalar_t v_th, 
    const scalar_t v_reset, 
    const int batch, 
    const int step, 
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = (idx_th - idx_th % dim) * step + (idx_th % dim);
    scalar_t potential = 0.0;

    for (int i = 0;i < step;++i, posi += dim) {
        potential = x[posi] + potential / tau;
        spike[posi] = potential >= v_th ? 1 : 0;
        psps[posi] = potential;
        potential = potential >= v_th ? v_reset : potential;
    }

}

template<typename scalar_t> __global__ void Leaky_integrate_BP(
    const scalar_t * psps, 
    const scalar_t * grad_out, 
    scalar_t * grad_x,
    const scalar_t tau, 
    const scalar_t v_th, 
    const int batch, 
    const int step, 
    const int dim, 
    const int suro, 
    const scalar_t alpha
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = idx_th * step + (dim - idx_th % dim) * (step - 1);
    scalar_t du = 0.0, u = 0.0;

    for (int i = step - 1;i >= 0;--i, posi -= dim) {
        u = psps[posi];
        scalar_t over_th = u - v_th, sg = 0;
        
        __SWITCH_ON_SURO_FUNC__(suro)
        
        du = grad_out[posi] * sg + du * (1 - scalar_t(over_th >= 0) - u * sg) / tau;
        
        grad_x[posi] = du;
    }
}

template<typename scalar_t> __global__ void Leaky_integrate_detached_BP(
    const scalar_t * psps, 
    const scalar_t * grad_out, 
    scalar_t * grad_x,
    const scalar_t tau, 
    const scalar_t v_th, 
    const int batch, 
    const int step, 
    const int dim, 
    const int suro, 
    const scalar_t alpha
) {    
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;
    int posi = idx_th * step + (dim - idx_th % dim) * (step - 1);
    scalar_t du = 0.0, u = 0.0;

    for (int i = step - 1;i >= 0;--i, posi -= dim) {
        u = psps[posi];
        scalar_t over_th = u - v_th, sg = 0;
        
        __SWITCH_ON_SURO_FUNC__(suro)
        
        du = over_th < 0 ? grad_out[posi] * sg + du / tau : grad_out[posi] * sg;
        
        grad_x[posi] = du;
    }
}

template<typename scalar_t> __global__ void srm_forward(
    const scalar_t * inputs, 
    scalar_t * spike, 
    scalar_t * delta_ut, 
    scalar_t * delta_u,
    const scalar_t taum, 
    const scalar_t taus, 
    const scalar_t e_taug, 
    const scalar_t threshold, 
    const int batch, 
    const int step, 
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;

    scalar_t first = 0, second = 0, grad = 0, u = 0, u_last = 0, ini = 0;
    int posi = idx_th % dim;
    
    scalar_t e_taum = 1. - 1. / taum;
    scalar_t e_taus = 1. - 1. / taus; 
    scalar_t coefficient = taum / (taum - taus); 

    posi = (idx_th - posi) * step + posi;

    for (int i = 0;i < step; ++i, posi += dim) {
        
        ini = inputs[posi];
        first = (ini + first) * e_taum;
        second = (ini + second) * e_taus;

        grad = (ini + grad) * e_taug;
        
        u = coefficient * (first - second);

        delta_u[posi] = u - u_last;
        delta_ut[posi] = grad;

        if (u > threshold) {
            spike[posi] = 1.0;
            u = u_last = grad = first = second = 0;
        }
        else {
            u_last = u;
            spike[posi] = 0.0;
        }
    }
}

template<typename scalar_t> __global__ void srm_backward(
    const scalar_t * grad_out, 
    const scalar_t * delta_ut, 
    const scalar_t * delta_u,
    const scalar_t * spike, 
    const scalar_t * epsw, 
    const scalar_t * epst, 
    scalar_t * grad_w, 
    scalar_t * grad_t, 
    const int batch, 
    const int step, 
    const int dim
) {
    int idx_th = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_th >= batch * dim) return;

    int posi = idx_th * step + (dim - idx_th % dim) * (step - 1);
    int delta_t = 0, out = 0;
    scalar_t gradt = 0, dut = 0, gradw = 0, duw = 0;

    for (int i = step - 1;i >= 0;--i, posi -= dim) {
        out = spike[posi];

        gradt = min(max(-1.0f / delta_ut[posi], -123456789.0f), 0.0f);
        gradw = min(max(-1.0f / delta_u[posi], -4.0f), 0.0f);

        dut = out ? gradt * grad_out[posi] : dut;
        duw = out ? gradw * grad_out[posi] : duw;

        delta_t = out ? 0 : delta_t + 1;

        grad_w[posi] = duw * epsw[delta_t];
        grad_t[posi] = dut * epst[delta_t];
    }
}