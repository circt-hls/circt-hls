typedef int in_int_t;
typedef int in_float_t;
typedef int inout_float_t;

void histogram(in_int_t feature[1000], in_float_t weight[1000], inout_float_t hist[1000], in_int_t n)
{
    for (int i = 0; i < n; ++i)
    {
        in_int_t m = feature[i];
        in_float_t wt = weight[i];
        inout_float_t x = hist[m];
        hist[m] = x + wt;
    }
}