#ifndef SHELL_SORT_H
#define SHELL_SORT_H

void shell_sort_test_orig(float *m, int n);
void shell_sort_half_orig(float *m, int n);
void shell_sort_hibbard_orig(float *m, int n);
void shell_sort_sedgewick_orig(float *m, int n);
void shell_sort_pratt_orig(float *m, int n);
void shell_sort_fib_orig(float *m, int n);

void shell_sort_opt_prepare();

void shell_sort_test_opt(float *m, int n);
void shell_sort_half_opt(float *m, int n);
void shell_sort_hibbard_opt(float *m, int n);
void shell_sort_sedgewick_opt(float *m, int n);
void shell_sort_pratt_opt(float *m, int n);
void shell_sort_fib_opt(float *m, int n);

#endif
