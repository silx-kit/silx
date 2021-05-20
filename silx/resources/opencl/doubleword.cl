/*
 * OpenCL library for double word floating point calculation using compensated arithmetics
 *
 * The theoritical basis can be found in Valentina Popescu's PhD thesis:
 * Towards fast and certified multi-precision libraries
 * Reference LYSEN036
 * http://www.theses.fr/2017LYSEN036
 * All page number and equation number are refering to this document. 
 * 
 * The precision of the calculation (bounds) is provided in ULP (smallest possible mantissa) 
 * and come from the table 2.2 (page 68 of the thesis).
 * The number of equivalent FLOP is taken from the table 2.3 (page 69 the thesis). 
 * Note that FLOP are not all equal: a division is much more expensive than an addition.
 */

//This library can be expanded to double-double by redefining fp, fp2 and one to double, double2 and 1.0.
#ifdef DOUBLEDOUBLE
#define fp double
#define fp2 double2
#define one 1.0
#else
#define fp float
#define fp2 float2
#define one 1.0f
#endif

/* Nota: i386 computer use x87 registers which are larger than the 32bits precision
 * which can invalidate the error compensation mechanism.
 *
 * We use the trick to declare some variable "volatile" to enforce the actual
 * precision reduction of those variables.
*/

#ifndef X87_VOLATILE
# define X87_VOLATILE
#endif

//Algorithm 1, p23, theorem 1.1.12. Requires e_x > e_y, valid if |x| > |y| 
inline fp2 fast_fp_plus_fp(fp x, fp y){
    X87_VOLATILE fp s = x + y;
    X87_VOLATILE fp z = s - x;
    fp e = y - z;
    return (fp2)(s, e);
}

//Algorithm 2, p24, same as fast_fp_plus_fp without the condition on e_x and e_y
inline fp2 fp_plus_fp(fp x, fp y){
    X87_VOLATILE fp s = x + y;
    X87_VOLATILE fp xp = s - y;
    X87_VOLATILE fp yp = s - xp;
    X87_VOLATILE fp dx = x - xp;
    X87_VOLATILE fp dy = y - yp;
    return (fp2)(s, dx+dy);  
}

//Algorithm 3, p24: multiplication with a FMA
inline fp2 fp_times_fp(fp x, fp y){
    fp p = x * y;
    fp e = fma(x, y, -p);
    return (fp2)(p, e);  
}

//Algorithm 7, p38: Addition of a FP to a DW. 10flop bounds:2u²+5u³
inline fp2 dw_plus_fp(fp2 x, fp y){
    fp2 s = fp_plus_fp(x.s0, y);
    X87_VOLATILE fp v = x.s1 + s.s1;
    return fast_fp_plus_fp(s.s0, v);
}

//Algorithm 9, p40: addition of two DW: 20flop bounds:3u²+13u³
inline fp2 dw_plus_dw(fp2 x, fp2 y){
    fp2 s = fp_plus_fp(x.s0, y.s0);
    fp2 t = fp_plus_fp(x.s1, y.s1);
    fp2 v = fast_fp_plus_fp(s.s0, s.s1 + t.s0);
    return fast_fp_plus_fp(v.s0, t.s1 + v.s1);
}

//Algorithm 12, p49: Multiplication FP*DW: 6flops bounds:2u²
inline fp2 dw_times_fp(fp2 x, fp y){
    fp2 c = fp_times_fp(x.s0, y);
    return fast_fp_plus_fp(c.s0, fma(x.s1, y, c.s1));
}

//Algorithm 14, p52: Multiplication DW*DW, 8 flops bounds:6u²
inline fp2 dw_times_dw(fp2 x, fp2 y){
    fp2 c = fp_times_fp(x.s0, y.s0);
    X87_VOLATILE fp l = fma(x.s1, y.s0, x.s0 * y.s1);
    return fast_fp_plus_fp(c.s0, c.s1 + l);
}

//Algorithm 17, p55: Division DW / FP, 10flops bounds: 3.5u²
inline fp2 dw_div_fp(fp2 x, fp y){
    X87_VOLATILE fp th = x.s0 / y;
    fp2 pi = fp_times_fp(th, y);
    fp2 d = x - pi;
    X87_VOLATILE fp delta = d.s0 + d.s1;
    X87_VOLATILE fp tl = delta/y; 
    return fast_fp_plus_fp(th, tl);
}

//Derived from algorithm 20, p64: Inversion 1/ DW, 22 flops
inline fp2 inv_dw(fp2 y){
    X87_VOLATILE fp th = one/y.s0;
    X87_VOLATILE fp rh = fma(-y.s0, th, one);
    X87_VOLATILE fp rl = -y.s1 * th;
    fp2 e = fast_fp_plus_fp(rh, rl);
    fp2 delta = dw_times_fp(e, th);
    return dw_plus_fp(delta, th);
}
    
//Algorithm 20, p64: Division DW / DW, 30 flops: bounds:9.8u²
inline fp2 dw_div_dw(fp2 x, fp2 y){
    return dw_times_dw(x, inv_dw(y));
}

