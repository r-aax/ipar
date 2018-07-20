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
    float k;
    float d[16], u[16], p[16], c[16], cm[16], sh[16], st[16], s[16], pms[16];

    for (int i = 0; i < 16; i++)
    {
        // Init side values.
        if (um[i] >= 0.0)
        {
            d[i] = dl[i];
            u[i] = ul[i];
            p[i] = pl[i];
            c[i] = cl[i];
            k = 1.0;
        }
        else
        {
            d[i] = dr[i];
            u[i] = ur[i];
            p[i] = pr[i];
            c[i] = -cr[i];
            k = -1.0;
        }

        // 4 cases (values on the left side or on the right side).
        od[i] = d[i];
        ou[i] = u[i];
        op[i] = p[i];

        if (pm[i] <= p[i])
        {
            sh[i] = u[i] - c[i];

            if (sh[i] * k < 0.0)
            {
                cm[i] = c[i] * powf(pm[i] / p[i], G1);
                st[i] = um[i] - cm[i];

                if (st[i] * k < 0.0)
                {
                    od[i] = d[i] * powf(pm[i] / p[i], igama);
                    ou[i] = um[i];
                    op[i] = pm[i];
                }
                else
                {
                    ou[i] = G5 * (c[i] + G7 * u[i]);
                    od[i] = d[i] * powf(ou[i] / c[i], G4);
                    op[i] = p[i] * powf(ou[i] / c[i], G3);
                }                
            }
        }
        else
        {
            pms[i] = pm[i] / p[i];
            s[i] = u[i] - c[i] * sqrtf(G2 * pms[i] + G1);

            if (s[i] * k < 0.0)
            {
                od[i] = d[i] * (pms[i] + G6) / (pms[i] * G6 + 1.0);
                ou[i] = um[i];
                op[i] = pm[i];
            }            
        }
    } 
}

// \brief All samples.
void samples_opt(float *dls, float *uls, float *pls, float *cls,
                 float *drs, float *urs, float *prs, float *crs,
                 float *pms, float *ums,
                 float *ds, float *us, float *ps)
{
    float d, u, p;

    for (int i = 0; i < TESTS_COUNT; i += 16)
    {
        samples_16_opt(&dls[i], &uls[i], &pls[i], &cls[i],
                       &drs[i], &urs[i], &prs[i], &crs[i],
                       &pms[i], &ums[i],
                       &ds[i], &us[i], &ps[i]);
    }
}
