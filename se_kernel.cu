/******************************************************************************
* PROGRAM: se_kernel
* PURPOSE: This is a collection of functions which is intended to optimize the
* 	speed of seed extension step in BWA MEM procedure.
*
*
* NAME: Vuong Pham-Duy.
*		College student.
*       Faculty of Computer Science and Engineering.
*       Ho Chi Minh University of Technology, Viet Nam.
*       vuongpd95@gmail.com
*
* DATE: 5/10/2017
*
******************************************************************************/
#include "se_kernel.h"

extern "C" void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, \
		int l_query, const uint8_t *query, const mem_chain_t *c, mem_alnreg_v *av);

/* CUDA support function */
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "[M::%s] GPUassert: %s %s %d\n", __func__, \
			cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

void print_mem_info()
{
	size_t free_byte;
	size_t total_byte;
	gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	fprintf(stderr, "[M::%s] GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", \
		__func__, used_db/ONE_MBYTE, free_db/ONE_MBYTE, total_db/ONE_MBYTE);
}

/********************
 *** SW extension ***
 ********************/
__device__
bool check_active(int32_t h, int32_t e) {
	if(h != -1 && e != -1) return true;
	else return false;
}
__device__
void reset(int32_t *h, int32_t *e) {
	*h = -1;
	*e = -1;
}

__device__ bool warp_lock(int req){
  return ((__ffs(__ballot(req))) == ((threadIdx.x & 31) + 1));
}

__device__ int mLock = 0;

extern __shared__ int32_t container[];
__global__
void sw_kernel(int *d_max, int *d_max_j, int *d_max_i, int *d_max_ie, int *d_gscore, int *d_max_off, \
		int w, int oe_ins, int e_ins, int o_del, int e_del, int oe_del, int m, \
		int tlen, int qlen, int passes, int t_lastp, int h0, int zdrop, \
		int32_t *h, int8_t *qp, const uint8_t *target) {

	__shared__ int break_cnt;
	__shared__ int max;
	__shared__ int max_i;
	__shared__ int max_j;
	__shared__ int max_ie;
	__shared__ int gscore;
	__shared__ int max_off;
	__shared__ int out_h[WARP];
	__shared__ int out_e[WARP];

	bool blocked = true;
	int in_h, in_e;
	int i;
	int active_ts, beg, end;
	int32_t *se, *sh;
	int8_t *sqp;

	/* Initialize */
	if(threadIdx.x == 0) {
		max = h0;
		max_i = -1;
		max_j = -1;
		max_ie = -1;
		gscore = -1;
		max_off = 0;
		break_cnt = 0;
	}

	i = threadIdx.x;
	sh = container;
	se = (int32_t*)&sh[qlen + 1];
	sqp = (int8_t*)&se[qlen + 1];
	for(;;) {
		if(i < qlen + 1) {
			sh[i] = h[i];
			se[i] = 0;
		}
		// qlen > 1, m = 5, qlen * m always bigger than qlen + 1
		if(i < qlen * m) sqp[i] = qp[i];
		else break;
		i += WARP;
	}
	__syncthreads();
	for(int i = 0; i < passes; i++) {
		if(i == passes - 1) {
			if(threadIdx.x >= t_lastp) break;
			else active_ts = t_lastp;
		} else active_ts = WARP;
		reset(&in_h, &in_e);
		reset(&out_h[threadIdx.x], &out_e[threadIdx.x]);
		beg = 0; end = qlen;

		int t, row_i, f = 0, h1, m = 0, mj = -1;
		int8_t *q = &sqp[target[i] * qlen];
		row_i = i * WARP + threadIdx.x;
		// apply the band and the constraint (if provided)
		if (beg < i - w) beg = i - w;
		if (end > i + w + 1) end = i + w + 1;
		if (end > qlen) end = qlen;
		// reset input, output

		if (beg == 0) {
			h1 = h0 - (o_del + e_del * (row_i + 1));
			if (h1 < 0) h1 = 0;
		} else h1 = 0;

		__syncthreads();

		do {
			if(threadIdx.x == 0) {
				in_h = sh[beg];
				in_e = se[beg];
			} else {
				in_h = out_h[threadIdx.x - 1];
				in_e = out_e[threadIdx.x - 1];
				// if(threadIdx.x == 1)
				//	printf("beg = %d, in_h = %d, in_e = %d.\n", beg, in_h, in_e);
			}
			__syncthreads();
			if(check_active(in_h, in_e)) {
				int h; 											// get H(i-1,j-1) and E(i-1,j)
				if(threadIdx.x != active_ts - 1) out_h[threadIdx.x] = h1;
				else if(i != passes - 1) sh[beg] = h1; 			// set H(i,j-1) for the next row
				in_h = in_h? in_h + q[beg] : 0;					// separating H and M to disallow a cigar like
																// "100M3I3D20M"
				h = in_h > in_e? in_h : in_e;   				// e and f are guaranteed to be non-negative,
																// so h>=0 even if M<0
				h = h > f? h : f;
				h1 = h;											// save H(i,j) to h1 for the next column
				//if (beg < end) {
					mj = m > h? mj : beg; 						// record the position where max score is achieved
					m = m > h? m : h;   						// m is stored at eh[mj+1]
				//}
				t = in_h - oe_del;
				t = t > 0? t : 0;
				in_e -= e_del;
				in_e = in_e > t? in_e : t;   					// computed E(i+1,j)

				//if(beg >= end) {
				//	if(threadIdx.x != active_ts - 1) out_e[threadIdx.x] = 0;		// save E(i+1,j) for the next row
				//	else if(i != passes - 1) sh[beg] = 0;
				//} else {
					if(threadIdx.x != active_ts - 1) out_e[threadIdx.x] = in_e;	// save E(i+1,j) for the next row
					else if(i != passes - 1) sh[beg] = in_e;
				//}

				t = in_h - oe_ins;
				t = t > 0? t : 0;
				f -= e_ins;
				f = f > t? f : t;  	 							// computed F(i,j+1)

				reset(&in_h, &in_e);
				beg += 1;
			}
			__syncthreads();
		} while((beg < end)/* || (threadIdx.x != (active_ts - 1) && beg < end + 1)*/);

		if(threadIdx.x != active_ts - 1) {
			out_h[threadIdx.x] = h1;
			out_e[threadIdx.x] = 0;
		} else if(i != passes - 1) {
			sh[beg] = h1;
			se[beg] = 0;
		}

		__syncthreads();
		while(blocked) {
			if(0 == atomicCAS(&mLock, 0, 1)) {
				// critical section
				if(beg == qlen) {
					max_ie = gscore > out_h[threadIdx.x]? max_ie : row_i;
					gscore = gscore > out_h[threadIdx.x]? gscore : out_h[threadIdx.x];
				}
				atomicExch(&mLock, 0);
				blocked = false;
			}
		}

		if(m == 0) atomicAdd(&break_cnt, 1);
		__syncthreads();
		//if(break_cnt > 0) break;

		blocked = true;
		while(blocked) {
			if (break_cnt > 0) break;
			if(0 == atomicCAS(&mLock, 0, 1)) {
				if(m > max) {
					max = m, max_i = row_i, max_j = mj;
					max_off = max_off > abs(mj - row_i)? max_off : abs(mj - row_i);
				} else if (zdrop > 0) {
					if (i - max_i > mj - max_j) {
						if (max - m - ((row_i - max_i) - (mj - max_j)) * e_del > zdrop) break_cnt += 1;
					} else {
						if (max - m - ((mj - max_j) - (row_i - max_i)) * e_ins > zdrop) break_cnt += 1;
					}
				}
				atomicExch(&mLock, 0);
				blocked = false;
			}
		}
		//if (break_cnt > 0) break;
	}

	if(threadIdx.x == 0) {
		*d_max = max;
		*d_max_i = max_i;
		*d_max_j = max_j;
		*d_max_ie = max_ie;
		*d_gscore = gscore;
		*d_max_off = max_off;
		printf("d_max = %d, d_max_i = %d, d_max_j = %d, d_max_ie = %d, d_gscore = %d, d_max_off = %d\n",\
				*d_max, *d_max_i, *d_max_j, *d_max_ie, *d_gscore, *d_max_off);
	}
}
int cuda_ksw_extend2(int qlen, const uint8_t *query, \
		int tlen, const uint8_t *target, \
		int m, const int8_t *mat, \
		int o_del, int e_del, int o_ins, \
		int e_ins, int w, int end_bonus, \
		int zdrop, int h0, int *_qle, \
		int *_tle, int *_gtle, int *_gscore, int *_max_off)
{
	int32_t *h;
	int8_t *qp; // query profile
	int i, j, k;
	int	oe_del = o_del + e_del; // opening and ending deletion
	int	oe_ins = o_ins + e_ins; // opening and ending insertion
	int max, max_i, max_j, max_ins, max_del, max_ie, gscore, max_off;
	int passes, t_lastp; // number of passes and number of thread active in the last pass
	assert(h0 > 0);
	// allocate memory
	qp = (int8_t*)malloc(qlen * m);
	h = (int32_t*)calloc(qlen + 1, sizeof(int32_t));

	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	// fill the first row
	h[0] = h0; h[1] = h0 > oe_ins? h0 - oe_ins : 0;
	for (j = 2; j <= qlen && h[j-1] > e_ins; ++j) {
		h[j] = h[j - 1] - e_ins;
	}
	// adjust $w if it is too large
	k = m * m;
	for (i = 0, max = 0; i < k; ++i) // get the max score
		max = max > mat[i]? max : mat[i];
	max_ins = (int)((double)(qlen * max + end_bonus - o_ins) / e_ins + 1.);
	max_ins = max_ins > 1? max_ins : 1;
	w = w < max_ins? w : max_ins;
	max_del = (int)((double)(qlen * max + end_bonus - o_del) / e_del + 1.);
	max_del = max_del > 1? max_del : 1;
	w = w < max_del? w : max_del; // TODO: is this necessary?
	// DP loop
	max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1; max_off = 0;

	// Initialize
	// memset: max, max_j, max_i, max_ie, gscore, max_off -> GPU
	// kernel parameters:
	// value: w, oe_ins, e_ins, o_del, e_del, oe_del, tlen, qlen, passes, t_lastp, h0, zdrop
	// memcpy: e[...], h[...], qp[...], target[...]
	int *d_max, *d_max_j, *d_max_i, *d_max_ie, *d_gscore, *d_max_off;
	int32_t *d_h;
	int8_t *d_qp;
	uint8_t *d_target;

	passes = (int)((double)tlen / (double)WARP + 1.);
	t_lastp = tlen - (tlen / WARP) * WARP;

	// gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, FIXED_HEAP * ONE_MBYTE));
	// Allocate device memory
	gpuErrchk(cudaMalloc(&d_max, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_max_j, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_max_i, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_max_ie, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_gscore, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_max_off, sizeof(int)));

	gpuErrchk(cudaMalloc(&d_h, sizeof(int32_t) * (qlen + 1)));
	gpuErrchk(cudaMalloc(&d_qp, sizeof(int8_t) * qlen * m));
	gpuErrchk(cudaMalloc(&d_target, sizeof(uint8_t) * tlen));
	/* memset d_variables
	gpuErrchk(cudaMemset(d_max, h0, sizeof(int)));
	gpuErrchk(cudaMemset(d_max_j, -1, sizeof(int)));
	gpuErrchk(cudaMemset(d_max_i, -1, sizeof(int)));
	gpuErrchk(cudaMemset(d_max_ie, -1, sizeof(int)));
	gpuErrchk(cudaMemset(d_gscore, -1, sizeof(int)));
	gpuErrchk(cudaMemset(d_max_off, 0, sizeof(int)));
	*/
	// Transfer data to GPU
	gpuErrchk(cudaMemcpy(d_h, h, sizeof(int32_t) * (qlen + 1), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_qp, qp, sizeof(int8_t) * qlen * m, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_target, target, sizeof(uint8_t) * tlen, cudaMemcpyHostToDevice));
	// The kernel

	printf("Passes = %d, t_lastp = %d\n", passes, t_lastp);
	sw_kernel<<<1, WARP, 2 * (qlen + 1) * sizeof(int32_t) + qlen * m * sizeof(int8_t)>>>\
			(d_max, d_max_j, d_max_i, d_max_ie, d_gscore, d_max_off, \
			w, oe_ins, e_ins, o_del, e_del, oe_del, m, \
			tlen, qlen, passes, t_lastp, h0, zdrop, \
			d_h, d_qp, d_target);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Deallocate host variables
	free(h); free(qp);
	// Get the result back from kernel
	gpuErrchk(cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_j, d_max_j, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_i, d_max_i, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_ie, d_max_ie, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&gscore, d_gscore, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&max_off, d_max_off, sizeof(int), cudaMemcpyDeviceToHost));
	// Deallocate CUDA variables
	gpuErrchk(cudaFree(d_max_j));
	gpuErrchk(cudaFree(d_max_i));
	gpuErrchk(cudaFree(d_max_ie));
	gpuErrchk(cudaFree(d_gscore));
	gpuErrchk(cudaFree(d_max_off));
	gpuErrchk(cudaFree(d_max));
	gpuErrchk(cudaFree(d_h));
	gpuErrchk(cudaFree(d_qp));
	gpuErrchk(cudaFree(d_target));
	// Return results
	if (_qle) *_qle = max_j + 1;
	if (_tle) *_tle = max_i + 1;
	if (_gtle) *_gtle = max_ie + 1;
	if (_gscore) *_gscore = gscore;
	if (_max_off) *_max_off = max_off;
	printf("[GPU:] max_j = %d, max_i = %d, max_ie = %d, "
			"gscore = %d, max_off = %d\n", max_j, max_i, max_ie, gscore, max_off);
	return max;
}

static inline int cal_max_gap(const mem_opt_t *opt, int qlen)
{
	int l_del = (int)((double)(qlen * opt->a - opt->o_del) / opt->e_del + 1.);
	int l_ins = (int)((double)(qlen * opt->a - opt->o_ins) / opt->e_ins + 1.);
	int l = l_del > l_ins? l_del : l_ins;
	l = l > 1? l : 1;
	return l < opt->w<<1? l : opt->w<<1;
}

void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, \
		const uint8_t *query, const mem_chain_t *c, mem_alnreg_v *av) {

	int i, k, rid, max_off[2], aw[2]; // aw: actual bandwidth used in extension
	int64_t l_pac = bns->l_pac, rmax[2], tmp, max = 0;
	const mem_seed_t *s;
	uint8_t *rseq = 0;
	uint64_t *srt;

	if (c->n == 0) return;
	// get the max possible span
	rmax[0] = l_pac << 1;
	rmax[1] = 0;
	for (i = 0; i < c->n; ++i) {
		int64_t b, e;
		const mem_seed_t *t = &c->seeds[i];
		b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg));
		e = t->rbeg + t->len + ((l_query - t->qbeg - t->len) + cal_max_gap(opt, l_query - t->qbeg - t->len));
		rmax[0] = rmax[0] < b? rmax[0] : b;
		rmax[1] = rmax[1] > e? rmax[1] : e;
		if (t->len > max) max = t->len;
	}
	// rmax[0] = rmax[0] > 0? rmax[0] : 0;
	if(rmax[0] <= 0) rmax[0] = 0;
	// rmax[1] = rmax[1] < l_pac<<1? rmax[1] : l_pac<<1;
	if (rmax[1] >= (l_pac << 1)) rmax[1] = l_pac << 1;

	if (rmax[0] < l_pac && l_pac < rmax[1]) { // crossing the forward-reverse boundary; then choose one side
		// this works because all seeds are guaranteed to be on the same strand
		if (c->seeds[0].rbeg < l_pac) rmax[1] = l_pac;
		else rmax[0] = l_pac;
	}
	// retrieve the reference sequence
	rseq = bns_fetch_seq(bns, pac, &rmax[0], c->seeds[0].rbeg, &rmax[1], &rid);
	assert(c->rid == rid);

	srt = (uint64_t*)malloc(c->n * 8);
	for (i = 0; i < c->n; ++i)
		srt[i] = (uint64_t)c->seeds[i].score<<32 | i;
	ks_introsort_64(c->n, srt);
	for (k = c->n - 1; k >= 0; --k) {
		mem_alnreg_t *a;
		s = &c->seeds[(uint32_t)srt[k]];

		for (i = 0; i < av->n; ++i) { // test whether extension has been made before
			mem_alnreg_t *p = &av->a[i];
			int64_t rd;
			int qd, w, max_gap;
			if (s->rbeg < p->rb || s->rbeg + s->len > p->re || s->qbeg < p->qb || s->qbeg + s->len > p->qe)
				continue; // not fully contained
			if (s->len - p->seedlen0 > .1 * l_query) continue; // this seed may give a better alignment
			// qd: distance ahead of the seed on query; rd: on reference
			qd = s->qbeg - p->qb; rd = s->rbeg - p->rb;
			max_gap = cal_max_gap(opt, qd < rd? qd : rd); // the maximal gap allowed in regions ahead of the seed
			w = max_gap < p->w? max_gap : p->w; // bounded by the band width
			if (qd - rd < w && rd - qd < w) break; // the seed is "around" a previous hit
			// similar to the previous four lines, but this time we look at the region behind
			qd = p->qe - (s->qbeg + s->len); rd = p->re - (s->rbeg + s->len);
			max_gap = cal_max_gap(opt, qd < rd? qd : rd);
			w = max_gap < p->w? max_gap : p->w;
			if (qd - rd < w && rd - qd < w) break;
		}

		if (i < av->n) { // the seed is (almost) contained in an existing alignment; further testing is needed to confirm it is not leading to a different aln
			if (bwa_verbose >= 4)
				printf("** Seed(%d) [%ld;%ld,%ld] is almost contained in an existing alignment [%d,%d) <=> [%ld,%ld)\n",
					   k, (long)s->len, (long)s->qbeg, (long)s->rbeg, av->a[i].qb, av->a[i].qe, (long)av->a[i].rb, \
					   (long)av->a[i].re);
			for (i = k + 1; i < c->n; ++i) { // check overlapping seeds in the same chain
				const mem_seed_t *t;
				if (srt[i] == 0) continue;
				t = &c->seeds[(uint32_t)srt[i]];
				if (t->len < s->len * .95) continue; // only check overlapping if t is long enough;
				// TODO: more efficient by early stopping
				if (s->qbeg <= t->qbeg && s->qbeg + s->len - t->qbeg >= s->len>>2 && t->qbeg - s->qbeg != t->rbeg - s->rbeg)
					break;
				if (t->qbeg <= s->qbeg && t->qbeg + t->len - s->qbeg >= s->len>>2 && s->qbeg - t->qbeg != s->rbeg - t->rbeg)
					break;
			}
			if (i == c->n) { // no overlapping seeds; then skip extension
				srt[k] = 0; // mark that seed extension has not been performed
				continue;
			}
			if (bwa_verbose >= 4)
				printf("** Seed(%d) might lead to a different alignment even though it is contained. "
						"Extension will be performed.\n", k);
		}

		a = kv_pushp(mem_alnreg_t, *av);
		memset(a, 0, sizeof(mem_alnreg_t));
		a->w = aw[0] = aw[1] = opt->w;
		a->score = a->truesc = -1;
		a->rid = c->rid;

		if (bwa_verbose >= 4) err_printf("** ---> Extending from seed(%d) [%ld;%ld,%ld] @ %s <---\n", k, \
				(long)s->len, (long)s->qbeg, (long)s->rbeg, bns->anns[c->rid].name);
		if (s->qbeg) { // left extension
			uint8_t *rs, *qs;
			int qle, tle, gtle, gscore;
			qs = (uint8_t*)malloc(s->qbeg);
			for (i = 0; i < s->qbeg; ++i) qs[i] = query[s->qbeg - 1 - i];
			tmp = s->rbeg - rmax[0];
			rs = (uint8_t*)malloc(tmp);
			for (i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i];
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[0] = opt->w << i;
				if (bwa_verbose >= 4) {
					int j;
					printf("*** Left ref:   ");
					for (j = 0; j < tmp; ++j)
						putchar("ACGTN"[(int)rs[j]]);
					putchar('\n');
					printf("*** Left query: ");
					for (j = 0; j < s->qbeg; ++j)
						putchar("ACGTN"[(int)qs[j]]);
					putchar('\n');
				}
				a->score = cuda_ksw_extend2(s->qbeg, qs, tmp, rs, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, \
						aw[0], opt->pen_clip5, opt->zdrop, s->len * opt->a, &qle, &tle, &gtle, &gscore, &max_off[0]);
				if (bwa_verbose >= 4) {
					printf("*** Left extension: prev_score=%d; score=%d; bandwidth=%d; max_off_diagonal_dist=%d\n", \
							prev, a->score, aw[0], max_off[0]); fflush(stdout);
				}
				if (a->score == prev || max_off[0] < (aw[0]>>1) + (aw[0]>>2)) break;
			}
			// check whether we prefer to reach the end of the query
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip5) { // local extension
				a->qb = s->qbeg - qle, a->rb = s->rbeg - tle;
				a->truesc = a->score;
			} else { // to-end extension
				a->qb = 0, a->rb = s->rbeg - gtle;
				a->truesc = gscore;
			}
			free(qs); free(rs);
		} else a->score = a->truesc = s->len * opt->a, a->qb = 0, a->rb = s->rbeg;

		if (s->qbeg + s->len != l_query) { // right extension
			int qle, tle, qe, re, gtle, gscore, sc0 = a->score;
			qe = s->qbeg + s->len;
			re = s->rbeg + s->len - rmax[0];
			assert(re >= 0);
			for (i = 0; i < MAX_BAND_TRY; ++i) {
				int prev = a->score;
				aw[1] = opt->w << i;
				if (bwa_verbose >= 4) {
					int j;
					printf("*** Right ref:   ");
					for (j = 0; j < rmax[1] - rmax[0] - re; ++j)
						putchar("ACGTN"[(int)rseq[re+j]]);
					putchar('\n');
					printf("*** Right query: ");
					for (j = 0; j < l_query - qe; ++j)
						putchar("ACGTN"[(int)query[qe+j]]);
					putchar('\n');
				}
				a->score = cuda_ksw_extend2(l_query - qe, query + qe, rmax[1] - rmax[0] - re, rseq + re, 5, opt->mat, \
						opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, aw[1], opt->pen_clip3, opt->zdrop, sc0, \
						&qle, &tle, &gtle, &gscore, &max_off[1]);
				if (bwa_verbose >= 4) {
					printf("*** Right extension: prev_score=%d; score=%d; bandwidth=%d; max_off_diagonal_dist=%d\n", prev, a->score, aw[1], max_off[1]);
					fflush(stdout);
				}
				if (a->score == prev || max_off[1] < (aw[1]>>1) + (aw[1]>>2)) break;
			}
			// similar to the above
			if (gscore <= 0 || gscore <= a->score - opt->pen_clip3) { // local extension
				a->qe = qe + qle, a->re = rmax[0] + re + tle;
				a->truesc += a->score - sc0;
			} else { // to-end extension
				a->qe = l_query, a->re = rmax[0] + re + gtle;
				a->truesc += gscore - sc0;
			}
		} else a->qe = l_query, a->re = s->rbeg + s->len;
		if (bwa_verbose >= 4)
			printf("*** Added alignment region: [%d,%d) <=> [%ld,%ld); score=%d; {left,right}_bandwidth={%d,%d}\n", \
					a->qb, a->qe, (long)a->rb, (long)a->re, a->score, aw[0], aw[1]);

		// compute seedcov
		for (i = 0, a->seedcov = 0; i < c->n; ++i) {
			const mem_seed_t *t = &c->seeds[i];
			// seed fully contained
			if (t->qbeg >= a->qb && t->qbeg + t->len <= a->qe && t->rbeg >= a->rb && t->rbeg + t->len <= a->re)
				a->seedcov += t->len; // this is not very accurate, but for approx. mapQ, this is good enough
		}
		a->w = aw[0] > aw[1]? aw[0] : aw[1];
		a->seedlen0 = s->len;

		a->frac_rep = c->frac_rep;
	}
	free(srt); free(rseq);
}
