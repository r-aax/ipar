/// \file
/// \brief Riemann solver.
///
/// Exact Riemann solver for the Euler equations in one dimension
/// Translated from the Fortran code er1pex.f and er1pex.ini
/// by Dr. E.F. Toro downloaded from
/// http://www.numeritek.com/numerica_software.html#freesample

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "riemann.h"
using namespace std;

#ifdef INTEL
#include <immintrin.h>
#endif

/// \brief
///
/// Purpose is to sample the solution throughout the wave
/// pattern. Pressure pm and velocity um in the
/// star region are known. Sampling is performed
/// in terms of the 'speed' s = x/t. Sampled
/// values are d, u, p.
///
/// TODO:
static void sample_orig(float dl, float ul, float pl, float cl,
                        float dr, float ur, float pr, float cr,
                        const float pm, const float um,
                        float &d, float &u, float &p)
{
    float c, cml, cmr, pml, pmr, shl, shr, sl, sr, stl, str;

    if (0.0 <= um)
    {
        // Sampling point lies to the left of the contact discontinuity.
        if (pm <= pl)
        {
            // Left rarefaction.
            shl = ul - cl;

            if (0.0 <= shl)
            {
                // Sampled point is left data state.
                d = dl;
                u = ul;
                p = pl;
            }
            else
            {
                cml = cl * pow(pm / pl, G1);
                stl = um - cml;

                if (0.0 > stl)
                {
                    // Sampled point is star left state.
                    d = dl * pow(pm / pl, 1.0 / GAMA);
                    u = um;
                    p = pm;
                }
                else
                {
                    // Sampled point is inside left fan.
                    u = G5 * (cl + G7 * ul);
                    c = G5 * (cl + G7 * ul);
                    d = dl * pow(c / cl, G4);
                    p = pl * pow(c / cl, G3);
                }
            }
        }
        else
        {
            // Left shock.
            pml = pm / pl;
            sl = ul - cl * sqrt(G2 * pml + G1);

            if (0.0 <= sl)
            {
                // Sampled point is left data state.
                d = dl;
                u = ul;
                p = pl;
            }
            else
            {
                // Sampled point is star left state.
                d = dl * (pml + G6) / (pml * G6 + 1.0);
                u = um;
                p = pm;
            }
        }
    }
    else
    {
        // Sampling point lies to the right of the contact discontinuity.
        if (pm > pr)
        {
            // Right shock.
            pmr = pm / pr;
            sr  = ur + cr * sqrt(G2 * pmr + G1);

            if (0.0 >= sr)
            {
                // Sampled point is right data state.
                d = dr;
                u = ur;
                p = pr;
            }
            else
            {
                // Sampled point is star right state.
                d = dr * (pmr + G6) / (pmr * G6 + 1.0);
                u = um;
                p = pm;
            }
        }
        else
        {
            // Right rarefaction.
            shr = ur + cr;
            if (0.0 >= shr)
            {
                // Sampled point is right data state.
                d = dr;
                u = ur;
                p = pr;
            }
            else
            {
                cmr = cr * pow(pm / pr, G1);
                str = um + cmr;

                if (0.0 <= str)
                {
                    // Sampled point is star right state.
                    d = dr * pow(pm / pr, 1.0 / GAMA);
                    u = um;
                    p = pm;
                }
                else
                {
                    // Sampled point is inside left fan.
                    u = G5 * (-cr + G7 * ur);
                    c = G5 * (cr - G7 * ur);
                    d = dr * pow(c / cr, G4);
                    p = pr * pow(c / cr, G3);
                }
            }
        }
    }
}

/// \brief
///
/// Purpose is to sample the solution throughout the wave
/// pattern. Pressure pm and velocity um in the
/// star region are known. Sampling is performed
/// in terms of the 'speed' s = x/t. Sampled
/// values are d, u, p.
///
/// TODO:
static void sample_opt(float dl, float ul, float pl, float cl,
                       float dr, float ur, float pr, float cr,
                       const float pm, const float um,
                       float &d, float &u, float &p)
{
    float c, cml, cmr, pml, pmr, shl, shr, sl, sr, stl, str;

    if (0.0 <= um)        
    {
        // 2 cases.
        d = dl;
        u = ul;
        p = pl;

        // Sampling point lies to the left of the contact discontinuity.
        if (pm <= pl)
        {
            // Left rarefaction.
            shl = ul - cl;

            if (!(0.0 <= shl))
            {
                cml = cl * pow(pm / pl, G1);
                stl = um - cml;

                if (0.0 > stl)
                {
                    // Sampled point is star left state.
                    d = dl * pow(pm / pl, 1.0 / GAMA);
                    u = um;
                    p = pm;
                }
                else
                {
                    // Sampled point is inside left fan.
                    u = G5 * (cl + G7 * ul);
                    c = G5 * (cl + G7 * ul);
                    d = dl * pow(c / cl, G4);
                    p = pl * pow(c / cl, G3);
                }
            }
        }
        else
        {
            // Left shock.
            pml = pm / pl;
            sl = ul - cl * sqrt(G2 * pml + G1);

            if (!(0.0 <= sl))
            {
                // Sampled point is star left state.
                d = dl * (pml + G6) / (pml * G6 + 1.0);
                u = um;
                p = pm;
            }
        }
    }
    else
    {
        // 2 cases.
        d = dr;
        u = ur;
        p = pr;

        // Sampling point lies to the right of the contact discontinuity.
        if (pm > pr)
        {
            // Right shock.
            pmr = pm / pr;
            sr  = ur + cr * sqrt(G2 * pmr + G1);

            if (!(0.0 >= sr))
            {
                // Sampled point is star right state.
                d = dr * (pmr + G6) / (pmr * G6 + 1.0);
                u = um;
                p = pm;
            }
        }
        else
        {
            // Right rarefaction.
            shr = ur + cr;

            if (!(0.0 >= shr))
            {
                cmr = cr * pow(pm / pr, G1);
                str = um + cmr;

                if (0.0 <= str)
                {
                    // Sampled point is star right state.
                    d = dr * pow(pm / pr, 1.0 / GAMA);
                    u = um;
                    p = pm;
                }
                else
                {
                    // Sampled point is inside left fan.
                    u = G5 * (-cr + G7 * ur);
                    c = G5 * (cr - G7 * ur);
                    d = dr * pow(c / cr, G4);
                    p = pr * pow(c / cr, G3);
                }
            }
        }
    }
}

// \brief All samples.
void samples_orig(float *dls, float *uls, float *pls, float *cls,
                  float *drs, float *urs, float *prs, float *crs,
                  float *pms, float *ums,
                  float *ds, float *us, float *ps)
{
    float d, u, p;

    for (int i = 0; i < TESTS_COUNT; i++)
    {
        sample_orig(dls[i], uls[i], pls[i], cls[i],
                    drs[i], urs[i], prs[i], crs[i],
                    pms[i], ums[i],
                    d, u, p);
        ds[i] = d;
        us[i] = u;
        ps[i] = p;
    }
}

// \brief 16 samples.
static void samples_16_opt(float *dl, float *ul, float *pl, float *cl,
                           float *dr, float *ur, float *pr, float *cr,
                           float *pm, float *um,
                           float *od, float *ou, float *op)
{
    float igama = 1.0 / GAMA;
    float ouc;
    float d[16], u[16], p[16], c[16], sh[16], st[16], s[16], pms[16], ums[16];
    int m[16];

    for (int i = 0; i < 16; i++)
    {
        m[i] = 0;
    }

    for (int i = 0; i < 16; i++)
    {
        // Init side values.
        if (um[i] >= 0.0)
        {
            d[i] = dl[i];
            u[i] = ul[i];
            p[i] = pl[i];
            c[i] = cl[i];
            ums[i] = um[i];
        }
        else
        {
            d[i] = dr[i];
            u[i] = -ur[i];
            p[i] = pr[i];
            c[i] = cr[i];
            ums[i] = -um[i];
        }

        // 4 cases (values on the left side or on the right side).
        od[i] = d[i];
        ou[i] = u[i];
        op[i] = p[i];

        pms[i] = pm[i] / p[i];
        sh[i] = u[i] - c[i];
        st[i] = ums[i] - c[i] * powf(pms[i], G1);
        s[i] = u[i] - c[i] * sqrtf(G2 * pms[i] + G1);

        if (pm[i] <= p[i])
        {
            if (sh[i] < 0.0)
            {
                if (st[i] < 0.0)
                {
                    od[i] = d[i] * powf(pms[i], igama);
                    ou[i] = ums[i];
                    op[i] = pm[i];
                }
                else
                {
                    m[i] = 1;
                }                
            }
        }
        else
        {
            if (s[i] < 0.0)
            {
                od[i] = d[i] * (pms[i] + G6) / (pms[i] * G6 + 1.0);
                ou[i] = ums[i];
                op[i] = pm[i];
            }            
        }
    }

    // Low prob - ignnore it.
    for (int i = 0; i < 16; i++)
    {
        if (m[i] == 1)
        {
            ou[i] = G5 * (c[i] + G7 * u[i]);
            ouc = ou[i] / c[i];
            od[i] = d[i] * powf(ouc, G4);
            op[i] = p[i] * powf(ouc, G3);
        }
    } 

    for (int i = 0; i < 16; i++)
    {
        if (um[i] < 0.0)
        {
            ou[i] = -ou[i];
        }
    }
}

// \brief All samples.
void samples_opt(float * __restrict__ dl,
                 float * __restrict__ ul,
                 float * __restrict__ pl,
                 float * __restrict__ cl,
                 float * __restrict__ dr,
                 float * __restrict__ ur,
                 float * __restrict__ pr,
                 float * __restrict__ cr,
                 float * __restrict__ pm,
                 float * __restrict__ um,
                 float * __restrict__ od,
                 float * __restrict__ ou,
                 float * __restrict__ op)
{

#ifdef INTEL

#define LD(ADDR) _mm512_load_ps(ADDR)
#define ST(ADDR, VAL) _mm512_store_ps(ADDR, VAL)

    __assume_aligned(dl, 64);
    __assume_aligned(ul, 64);
    __assume_aligned(pl, 64);
    __assume_aligned(cl, 64);
    __assume_aligned(dr, 64);
    __assume_aligned(ur, 64);
    __assume_aligned(pr, 64);
    __assume_aligned(cr, 64);
    __assume_aligned(pm, 64);
    __assume_aligned(um, 64);
    __assume_aligned(od, 64);
    __assume_aligned(ou, 64);
    __assume_aligned(op, 64);

    // Constants.
    __m512 z = _mm512_setzero_ps();
    __m512 v1 = _mm512_set1_ps(1.0);
    __m512 g1 = _mm512_set1_ps(G1);
    __m512 g2 = _mm512_set1_ps(G2);
    __m512 g3 = _mm512_set1_ps(G3);
    __m512 g4 = _mm512_set1_ps(G4);
    __m512 g5 = _mm512_set1_ps(G5);
    __m512 g6 = _mm512_set1_ps(G6);
    __m512 g7 = _mm512_set1_ps(G7);
    __m512 igama = _mm512_set1_ps(1.0 / GAMA);

    __m512 d, u, p, c, v_pm, v_um, ums, pms, sh, st, s, tmp, v_od, v_ou, v_op;
    __mmask16 cond_um, cond_pm, cond_sh, cond_st_1, cond_st_2, cond_s;

    for (int j = 0; j < TESTS_COUNT; j += 16)
    {
	// Load.
	v_pm = LD(&pm[j]);
	v_um = LD(&um[j]);

	// d/u/p/c/ums
	cond_um = _mm512_cmp_ps_mask(v_um, z, _MM_CMPINT_LT);
	d = _mm512_mask_blend_ps(cond_um, LD(&dl[j]), LD(&dr[j]));
	u = _mm512_mask_blend_ps(cond_um, LD(&ul[j]), LD(&ur[j]));
	p = _mm512_mask_blend_ps(cond_um, LD(&pl[j]), LD(&pr[j]));
	c = _mm512_mask_blend_ps(cond_um, LD(&cl[j]), LD(&cr[j]));
	ums = LD(&um[j]);
	u = _mm512_mask_sub_ps(u, cond_um, z, u);
	ums = _mm512_mask_sub_ps(ums, cond_um, z, ums);

	// 4 branches.
	v_od = d;
	v_ou = u;
	v_op = p;

	// Calculate main values.
	pms = _mm512_div_ps(v_pm, p);
	sh = _mm512_sub_ps(u, c);
	st = _mm512_fnmadd_ps(_mm512_pow_ps(pms, g1), c, ums);
	s = _mm512_fnmadd_ps(c, _mm512_sqrt_ps(_mm512_fmadd_ps(g2, pms, g1)), u);

	// Conditions.
	cond_pm = _mm512_cmp_ps_mask(v_pm, p, _MM_CMPINT_LE);
	cond_sh = _mm512_mask_cmp_ps_mask(cond_pm, sh, z, _MM_CMPINT_LT);
	cond_st_1 = _mm512_mask_cmp_ps_mask(cond_sh, st, z, _MM_CMPINT_LT);
	cond_st_2 = _mm512_mask_cmp_ps_mask(cond_sh, st, z, _MM_CMPINT_GE);
	cond_s = _mm512_mask_cmp_ps_mask(~cond_pm, s, z, _MM_CMPINT_LT);
	
	// Store.
	tmp = _mm512_pow_ps(pms, igama);
	v_od = _mm512_mask_mov_ps(v_od, cond_st_1, _mm512_mul_ps(d, tmp));
	tmp = _mm512_fmadd_ps(pms, g6, v1);
	tmp = _mm512_div_ps(_mm512_add_ps(pms, g6), tmp);
	v_od = _mm512_mask_mov_ps(v_od, cond_s, _mm512_mul_ps(d, tmp));
	v_ou = _mm512_mask_mov_ps(v_ou, cond_st_1 | cond_s, ums);
	v_op = _mm512_mask_mov_ps(v_op, cond_st_1 | cond_s, v_pm);

        // Low prob - ignnore it.
        if (cond_st_2)
        {
	    tmp = _mm512_fmadd_ps(g7, u, c);
	    v_ou = _mm512_mask_mov_ps(v_ou, cond_st_2, _mm512_mul_ps(g5, tmp));
	    __m512 v_ouc = _mm512_div_ps(v_ou, c);
	    tmp = _mm512_pow_ps(v_ouc, g4);
	    v_od = _mm512_mask_mov_ps(v_od, cond_st_2, _mm512_mul_ps(d, tmp));
	    tmp = _mm512_pow_ps(v_ouc, g3);
	    v_op = _mm512_mask_mov_ps(v_op, cond_st_2, _mm512_mul_ps(p, tmp));
	}

	// Final store.
	v_ou = _mm512_mask_sub_ps(v_ou, cond_um, z, v_ou);
	ST(&od[j], v_od);
	ST(&ou[j], v_ou);
	ST(&op[j], v_op);
    }

#else

    float igama = 1.0 / GAMA;
    float ouc;
    float d[16], u[16], p[16], c[16], sh[16], st[16], s[16], pms[16], ums[16];
    int m[16];

    for (int j = 0; j < TESTS_COUNT; j += 16)
    {
        for (int i = 0; i < 16; i++)
        {
            m[i] = 0;
        }

        for (int i = 0; i < 16; i++)
        {
            // Init side values.
            if (um[j + i] >= 0.0)
            {
                d[i] = dl[j + i];
                u[i] = ul[j + i];
                p[i] = pl[j + i];
                c[i] = cl[j + i];
                ums[i] = um[j + i];
            }
            else
            {
                d[i] = dr[j + i];
                u[i] = -ur[j + i];
                p[i] = pr[j + i];
                c[i] = cr[j + i];
                ums[i] = -um[j + i];
            }

            // 4 cases (values on the left side or on the right side).
            od[j + i] = d[i];
            ou[j + i] = u[i];
            op[j + i] = p[i];

            pms[i] = pm[j + i] / p[i];
            sh[i] = u[i] - c[i];
            st[i] = ums[i] - c[i] * powf(pms[i], G1);
            s[i] = u[i] - c[i] * sqrtf(G2 * pms[i] + G1);

            if (pm[j + i] <= p[i])
            {
                if (sh[i] < 0.0)
                {
                    if (st[i] < 0.0)
                    {
                        od[j + i] = d[i] * powf(pms[i], igama);
                        ou[j + i] = ums[i];
                        op[j + i] = pm[j + i];
                    }
                    else
                    {
                        m[i] = 1;
                    }                
                }
            }
            else
            {
                if (s[i] < 0.0)
                {
                    od[j + i] = d[i] * (pms[i] + G6) / (pms[i] * G6 + 1.0);
                    ou[j + i] = ums[i];
                    op[j + i] = pm[j + i];
                }            
            }
        }

        // Low prob - ignnore it.
        for (int i = 0; i < 16; i++)
        {
            if (m[i] == 1)
            {
                ou[j + i] = G5 * (c[i] + G7 * u[i]);
                ouc = ou[j + i] / c[i];
                od[j + i] = d[i] * powf(ouc, G4);
                op[j + i] = p[i] * powf(ouc, G3);
            }
        } 

        for (int i = 0; i < 16; i++)
        {
            if (um[j + i] < 0.0)
            {
                ou[j + i] = -ou[j + i];
            }
        }
    }

#endif

}
