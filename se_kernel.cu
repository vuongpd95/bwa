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
	/** Query GPU device **/
	int device, proc_cnt;
	cudaDeviceProp props;
	gpuErrchk(cudaGetDevice(&device));
	gpuErrchk(cudaGetDeviceProperties(&props, device));
	proc_cnt = props. multiProcessorCount;
	/** End querying **/
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
		/** Original
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
		**/
		/** CUDA ported **/
		int num_block;
		int tp_block;

		/** END of CUDA ported **/
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
				a->score = ksw_extend2(s->qbeg, qs, tmp, rs, 5, opt->mat, opt->o_del, opt->e_del, opt->o_ins, opt->e_ins, \
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
				a->score = ksw_extend2(l_query - qe, query + qe, rmax[1] - rmax[0] - re, rseq + re, 5, opt->mat, \
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
