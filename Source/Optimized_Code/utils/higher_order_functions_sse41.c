#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ViennaRNA/utils/basic.h"

#include <emmintrin.h>
#include <smmintrin.h>

static int
horizontal_min_Vec4i(__m128i x);


PUBLIC int
vrna_fun_zip_add_min_sse41(const int  *e1,
                           const int  *e2,
                           int        count)
{
  int     i       = 0;
  int     decomp  = INF;

  __m128i inf = _mm_set1_epi32(INF);

  for (i = 0; i < count - 3; i += 4) {
    __m128i a = _mm_loadu_si128((__m128i *)&e1[i]);
    __m128i b = _mm_loadu_si128((__m128i *)&e2[i]);
    __m128i c = _mm_add_epi32(a, b);

    /* create mask for non-INF values */
    __m128i mask = _mm_and_si128(_mm_cmplt_epi32(a, inf),
                                 _mm_cmplt_epi32(b, inf));

    /* delete results where a or b has been INF before */
    c = _mm_and_si128(mask, c);

    /* fill all values with INF if they've been INF in a or b before */
    __m128i   res = _mm_or_si128(c, _mm_andnot_si128(mask, inf));
    const int en  = horizontal_min_Vec4i(res);

    decomp = MIN2(decomp, en);
  }

  for (; i < count; i++) {
    if ((e1[i] != INF) && (e2[i] != INF)) {
      const int en = e1[i] + e2[i];
      decomp = MIN2(decomp, en);
    }
  }

  return decomp;
}


/*
 *  SSE minimum
 *  see also: http://stackoverflow.com/questions/9877700/getting-max-value-in-a-m128i-vector-with-sse
 */
static int
horizontal_min_Vec4i(__m128i x)
{
  __m128i min1  = _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 0, 3, 2));
  __m128i min2  = _mm_min_epi32(x, min1);
  __m128i min3  = _mm_shuffle_epi32(min2, _MM_SHUFFLE(0, 0, 0, 1));
  __m128i min4  = _mm_min_epi32(min2, min3);

  return _mm_cvtsi128_si32(min4);
}



#define MAX_NEW(i, j) (((i) > (j)) ? (i) : (j))

#define PERFORM_SSE(v_fmi_kmid, v_fm_k1jmid, v_add, v_dmli_j) 	\
        v_add       = _mm_add_epi32(v_fmi_kmid, v_fm_k1jmid);	\
       	v_dmli_j    = _mm_min_epi32(v_dmli_j, v_add);		\
     								\
       
#define PERFORM_TRANSPOSE_SSE(io0, io1, io2, io3, tmp0, tmp1, tmp2, tmp3) \
	tmp0 = _mm_unpacklo_epi32(io0, io1); 	\
	tmp1 = _mm_unpacklo_epi32(io2, io3);	\
	tmp2 = _mm_unpackhi_epi32(io0, io1);	\
	tmp3 = _mm_unpackhi_epi32(io2, io3);	\
						\
	io0 = _mm_unpacklo_epi64(tmp0, tmp1);	\
	io1 = _mm_unpackhi_epi64(tmp0, tmp1);	\
	io2 = _mm_unpacklo_epi64(tmp2, tmp3);	\
	io3 = _mm_unpackhi_epi64(tmp2, tmp3);	



PUBLIC int
vrna_add_min_vidit_sse41(vrna_fold_compound_t *fc,
                           int  ii,
                           int  jj,
                           int  tile,
			   int  **ha_fmi,
                           int  **ha_dmli,
			   int64_t flop_count)
{
  
  int             i, j, k, k1j, *indx, *fm;
  indx          = fc->jindx;
  fm            = fc->matrices->fML;

  int iin,jin,kin;


  int i_start = MAX_NEW(ii,0);
  int i_last  = MAX_NEW((ii-tile), 0);

  int k_start= (ii + 2);
  int k_last = (jj - tile - 3);
  int k_last_aligned = k_start + (((k_last-k_start) >> 2 ) << 2);

  int j_start = (jj-tile+1);
  int j_last   = (jj+1);
  int j_last_aligned = j_start + (((j_last-j_start) >> 2 ) << 2);

  int tile2 = tile+2;
  int kp1;
  int i2end = i_start % tile;
  int i_size = (i_start-i_last);
  int j_size = (j_last_aligned-j_start);
  int i_mat, j_mat;
  __m128i v_dml[i_size * j_size];

  for (i = i_start, i_mat = 0; i > i_last; i -= 1, i_mat += 1) {
    for (j = j_start, j_mat = 0; j < j_last_aligned; j += 4, j_mat += 1) {
      v_dml[i_mat + j_mat * i_size] = _mm_loadu_si128((__m128i*) &ha_dmli[i%tile2][j]);
    }
  }

  for (k = k_start; k < k_last_aligned; k+= 4) {
    kp1 = k+1;
    #pragma omp parallel for collapse(2)
    for (j = j_start, j_mat = 0; j < j_last_aligned; j += 4, j_mat += i_size) {

      __m128i v_fm_k1j_N0 = _mm_loadu_si128((__m128i*) &fm [ indx[j] + kp1 ]);
      __m128i v_fm_k1j_N1 = _mm_loadu_si128((__m128i*) &fm [ indx[j+1] + kp1 ]);
      __m128i v_fm_k1j_N2 = _mm_loadu_si128((__m128i*) &fm [ indx[j+2] + kp1 ]);
      __m128i v_fm_k1j_N3 = _mm_loadu_si128((__m128i*) &fm [ indx[j+3] + kp1 ]);

      {
        __m128i temp0, temp1, temp2, temp3;
        PERFORM_TRANSPOSE_SSE(v_fm_k1j_N0, v_fm_k1j_N1, v_fm_k1j_N2, v_fm_k1j_N3, temp0, temp1, temp2, temp3)
      }

      __m128i v_add;
      for (i = i2end, i_mat=j_mat; i >= 0; i -= 1, i_mat+=1) {

        int *fmi = ha_fmi[i];

	__m128i v_fmi_k0 = _mm_set1_epi32(fmi[k]);
	__m128i v_fmi_k1 = _mm_set1_epi32(fmi[kp1]);
	__m128i v_fmi_k2 = _mm_set1_epi32(fmi[k+2]);
	__m128i v_fmi_k3 = _mm_set1_epi32(fmi[k+3]);

	PERFORM_SSE(v_fmi_k0, v_fm_k1j_N0, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k1, v_fm_k1j_N1, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k2, v_fm_k1j_N2, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k3, v_fm_k1j_N3, v_add, v_dml[i_mat])

      }

      for (i=tile-1 ; i > i2end; i -= 1, i_mat+=1) {

        int *fmi = ha_fmi[i];

	__m128i v_fmi_k0 = _mm_set1_epi32(fmi[k]);
	__m128i v_fmi_k1 = _mm_set1_epi32(fmi[kp1]);
	__m128i v_fmi_k2 = _mm_set1_epi32(fmi[k+2]);
	__m128i v_fmi_k3 = _mm_set1_epi32(fmi[k+3]);

	PERFORM_SSE(v_fmi_k0, v_fm_k1j_N0, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k1, v_fm_k1j_N1, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k2, v_fm_k1j_N2, v_add, v_dml[i_mat])
	PERFORM_SSE(v_fmi_k3, v_fm_k1j_N3, v_add, v_dml[i_mat])

      }


    }
  }

  for (i = i_start, i_mat = 0; i > i_last; i -= 1, i_mat += 1) {
    for (j = j_start, j_mat = 0; j < j_last_aligned; j += 4, j_mat += 1) {
      _mm_storeu_si128((__m128i*) &ha_dmli[i%tile2][j], v_dml[i_mat + j_mat * i_size]);
    }
  }


  for (i = i_start; i > i_last; i -= 1) {
      int *fmi = ha_fmi[i%tile];
      int *dmli= ha_dmli[i%tile2];

      for (k = k_start; k < k_last_aligned; k+= 4) {
        for (kin = 0; kin < 4; kin += 1) {
	  for ( ; j < j_last; j += 1) {
              k1j   = indx[j] + k+kin + 1;
              if ((fmi[k+kin] != INF) && (fm[k1j] != INF)) {
                  dmli[j] = MIN2(dmli[j], (fmi[k+kin] + fm[k1j]) );
              }
          }
        }
      }

      for ( ; k < k_last; k+= 1) {
          for (j = j_start; j < j_last_aligned; j += 4) {
            for (jin = 0; jin < 4; jin += 1) {
              k1j   = indx[j+jin] + k + 1;
              if ((fmi[k] != INF) && (fm[k1j] != INF)) {
                  dmli[j+jin] = MIN2(dmli[j+jin], (fmi[k] + fm[k1j]) );
              }
            }
          }
      }

      for ( ; k < k_last; k+= 1) {
	  for ( ; j < j_last; j += 1) {
              k1j   = indx[j] + k + 1;
              if ((fmi[k] != INF) && (fm[k1j] != INF)) {
                  dmli[j] = MIN2(dmli[j], (fmi[k] + fm[k1j]) );
              }
          }
      }

  } 

  //return flop_count;
  return 0;
}

