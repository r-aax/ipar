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
                           float *d, float *u, float *p)
{
    float cml[16], cmr[16], pml[16], pmr[16], shl[16], shr[16], sl[16], sr[16], stl[16], str[16];
    float igama = 1.0 / GAMA;
    float d_side[16], u_side[16], p_side[16];

    for (int i = 0; i < 16; i++)
    {
        // Init side values.
        if (0.0 <= um[i])
        {
            d_side[i] = dl[i];
            u_side[i] = ul[i];
            p_side[i] = pl[i];
        }
        else
        {
            d_side[i] = dr[i];
            u_side[i] = ur[i];
            p_side[i] = pr[i];
        }

        // 4 cases (values on the left side or on the right side).
        d[i] = d_side[i];
        u[i] = u_side[i];
        p[i] = p_side[i];

        if (0.0 <= um[i])        
        {
            // Sampling point lies to the left of the contact discontinuity.
            if (pm[i] <= pl[i])
            {
                // Left rarefaction.
                shl[i] = ul[i] - cl[i];

                if (!(0.0 <= shl[i]))
                {
                    cml[i] = cl[i] * powf(pm[i] / pl[i], G1);
                    stl[i] = um[i] - cml[i];

                    if (0.0 > stl[i])
                    {
                        // Sampled point is star left state.
                        d[i] = dl[i] * powf(pm[i] / pl[i], igama);
                        u[i] = um[i];
                        p[i] = pm[i];
                    }
                    else
                    {
                        // Sampled point is inside left fan.
                        u[i] = G5 * (cl[i] + G7 * ul[i]);
                        d[i] = dl[i] * powf(u[i] / cl[i], G4);
                        p[i] = pl[i] * powf(u[i] / cl[i], G3);
                    }
                }
            }
            else
            {
                // Left shock.
                pml[i] = pm[i] / pl[i];
                sl[i] = ul[i] - cl[i] * sqrtf(G2 * pml[i] + G1);

                if (!(0.0 <= sl[i]))
                {
                    // Sampled point is star left state.
                    d[i] = dl[i] * (pml[i] + G6) / (pml[i] * G6 + 1.0);
                    u[i] = um[i];
                    p[i] = pm[i];
                }
            }
        }
        else
        {
            // Sampling point lies to the right of the contact discontinuity.
            if (pm[i] > pr[i])
            {
                // Right shock.
                pmr[i] = pm[i] / pr[i];
                sr[i]  = ur[i] + cr[i] * sqrtf(G2 * pmr[i] + G1);

                if (!(0.0 >= sr[i]))
                {
                    // Sampled point is star right state.
                    d[i] = dr[i] * (pmr[i] + G6) / (pmr[i] * G6 + 1.0);
                    u[i] = um[i];
                    p[i] = pm[i];
                }
            }
            else
            {
                // Right rarefaction.
                shr[i] = ur[i] + cr[i];

                if (!(0.0 >= shr[i]))
                {
                    cmr[i] = cr[i] * powf(pm[i] / pr[i], G1);
                    str[i] = um[i] + cmr[i];

                    if (0.0 <= str[i])
                    {
                        // Sampled point is star right state.
                        d[i] = dr[i] * powf(pm[i] / pr[i], igama);
                        u[i] = um[i];
                        p[i] = pm[i];
                    }
                    else
                    {
                        // Sampled point is inside left fan.
                        u[i] = G5 * (-cr[i] + G7 * ur[i]);
                        d[i] = dr[i] * powf(-u[i] / cr[i], G4);
                        p[i] = pr[i] * powf(-u[i] / cr[i], G3);
                    }
                }
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
