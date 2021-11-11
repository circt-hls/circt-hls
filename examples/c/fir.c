typedef int in_int_t;
typedef int inout_int_t;

int fir (in_int_t d_i[1000], in_int_t idx[1000] ) {
	int i;
	int tmp=0;

	for (i=0;i<1000;i++) {
		tmp += idx [i] * d_i[999-i];

	}
	return tmp;
}