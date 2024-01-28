#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ViennaRNA/utils/basic.h"

#include <immintrin.h>


PUBLIC int
vrna_fun_zip_add_min_avx512(const int *e1,
                            const int *e2,
                            int       count)
{
  int     i       = 0;
  int     decomp  = INF;

  __m512i inf = _mm512_set1_epi32(INF);

  /* WBL 21 Aug 2018 Add SSE512 code from sources_034_578/modular_decomposition_id3.c by hand */
  for (i = 0; i < count - 15; i += 16) {
    __m512i   a = _mm512_loadu_si512((__m512i *)&e1[i]);
    __m512i   b = _mm512_loadu_si512((__m512i *)&e2[i]);

    /* compute mask for entries where both, a and b, are less than INF */
    __mmask16 mask = _kand_mask16(_mm512_cmplt_epi32_mask(a, inf),
                                  _mm512_cmplt_epi32_mask(b, inf));

    /* add values */
    __m512i   c = _mm512_add_epi32(a, b);

    /* reduce to minimum (only those where one of the source values was not INF before) */
    const int en = _mm512_mask_reduce_min_epi32(mask, c);

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





#define MAX_NEW(i, j) (((i) > (j)) ? (i) : (j))


#define PERFORM_AVX2(v_fm_ik, v_fm_kj, v_add, v_out) 			\
    v_add = _mm256_add_epi32(v_fm_ik, _mm256_cvtps_epi32(v_fm_kj));	\
    v_out = _mm256_min_epi32(v_out, v_add);				\

 
#define PERFORM_TRANSPOSE_AVX2(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7) 	\
    _tmp0 = _mm256_unpacklo_ps(_r0, _r1); 				\
    _tmp1 = _mm256_unpackhi_ps(_r0, _r1); 				\
    _tmp2 = _mm256_unpacklo_ps(_r2, _r3); 				\
    _tmp3 = _mm256_unpackhi_ps(_r2, _r3); 				\
    _tmp4 = _mm256_unpacklo_ps(_r4, _r5); 				\
    _tmp5 = _mm256_unpackhi_ps(_r4, _r5); 				\
    _tmp6 = _mm256_unpacklo_ps(_r6, _r7); 				\
    _tmp7 = _mm256_unpackhi_ps(_r6, _r7); 				\
									\
    _tmp8 = _mm256_shuffle_ps(_tmp0,_tmp2, 0x4E); 			\
    _r0   = _mm256_blend_ps(_tmp0, _tmp8, 0xCC);			\
    _r1   = _mm256_blend_ps(_tmp2, _tmp8, 0x33);			\
    _tmp8 = _mm256_shuffle_ps(_tmp1,_tmp3, 0x4E);			\
    _r2   = _mm256_blend_ps(_tmp1, _tmp8, 0xCC);			\
    _r3   = _mm256_blend_ps(_tmp3, _tmp8, 0x33);			\
    _tmp8 = _mm256_shuffle_ps(_tmp4,_tmp6, 0x4E);			\
    _r4   = _mm256_blend_ps(_tmp4, _tmp8, 0xCC);			\
    _r5   = _mm256_blend_ps(_tmp6, _tmp8, 0x33);			\
    _tmp8 = _mm256_shuffle_ps(_tmp5,_tmp7, 0x4E);			\
    _r6   = _mm256_blend_ps(_tmp5, _tmp8, 0xCC);			\
    _r7   = _mm256_blend_ps(_tmp7, _tmp8, 0x33);			\


       
#define PERFORM_TRANSPOSE_AVX2_LO_MEM(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7) \
    _tmp0 = _mm256_unpacklo_ps(_r0, _r0); 					\
    _tmp1 = _mm256_unpackhi_ps(_r0, _r0); 					\
 										\
    _tmp2 = _mm256_shuffle_ps(_tmp0, _tmp0, _MM_SHUFFLE(1, 0, 1, 0)); 		\
    _tmp3 = _mm256_shuffle_ps(_tmp0, _tmp0, _MM_SHUFFLE(3, 2, 3, 2)); 		\
    _tmp4 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(1, 0, 1, 0)); 		\
    _tmp5 = _mm256_shuffle_ps(_tmp1, _tmp1, _MM_SHUFFLE(3, 2, 3, 2)); 		\
 										\
    _r0 = _mm256_permute2f128_ps(_tmp2, _tmp2, _MM_SHUFFLE(0, 2, 0, 0)); 	\
    _r1 = _mm256_permute2f128_ps(_tmp3, _tmp3, _MM_SHUFFLE(0, 2, 0, 0)); 	\
    _r2 = _mm256_permute2f128_ps(_tmp4, _tmp4, _MM_SHUFFLE(0, 2, 0, 0)); 	\
    _r3 = _mm256_permute2f128_ps(_tmp5, _tmp5, _MM_SHUFFLE(0, 2, 0, 0)); 	\
    _r4 = _mm256_permute2f128_ps(_tmp2, _tmp2, _MM_SHUFFLE(0, 3, 0, 1)); 	\
    _r5 = _mm256_permute2f128_ps(_tmp3, _tmp3, _MM_SHUFFLE(0, 3, 0, 1)); 	\
    _r6 = _mm256_permute2f128_ps(_tmp4, _tmp4, _MM_SHUFFLE(0, 3, 0, 1)); 	\
    _r7 = _mm256_permute2f128_ps(_tmp5, _tmp5, _MM_SHUFFLE(0, 3, 0, 1)); 	\


 
#define PERFORM_TRANSPOSE_AVX2_LO_MEM_FAST(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7) 	\
    _tmp0 = _mm256_unpacklo_ps(_r0, _r0); 				\
    _tmp1 = _mm256_unpackhi_ps(_r0, _r0); 				\
    _tmp2 = _mm256_unpacklo_ps(_r4, _r4); 				\
    _tmp3 = _mm256_unpackhi_ps(_r4, _r4); 				\
									\
    _tmp4 = _mm256_shuffle_ps(_tmp0,_tmp0, 0x4E); 			\
    _r0   = _mm256_blend_ps(_tmp0, _tmp4, 0xCC);			\
    _r1   = _mm256_blend_ps(_tmp0, _tmp4, 0x33);			\
    _tmp4 = _mm256_shuffle_ps(_tmp1,_tmp1, 0x4E);			\
    _r2   = _mm256_blend_ps(_tmp1, _tmp4, 0xCC);			\
    _r3   = _mm256_blend_ps(_tmp1, _tmp4, 0x33);			\
    _tmp4 = _mm256_shuffle_ps(_tmp2,_tmp2, 0x4E);			\
    _r4   = _mm256_blend_ps(_tmp2, _tmp4, 0xCC);			\
    _r5   = _mm256_blend_ps(_tmp2, _tmp4, 0x33);			\
    _tmp4 = _mm256_shuffle_ps(_tmp3,_tmp3, 0x4E);			\
    _r6   = _mm256_blend_ps(_tmp3, _tmp4, 0xCC);			\
    _r7   = _mm256_blend_ps(_tmp3, _tmp4, 0x33);			\




PUBLIC int64_t
vrna_add_min_new_avx2(vrna_fold_compound_t *fc,
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

  int jin,kin;


  int i_start = MAX_NEW(ii,0);
  int i_last  = MAX_NEW((ii-tile), 0);

  int k_start= (ii + 2);
  int k_last = (jj - tile - 3);
  int k_last_aligned = k_start + (((k_last-k_start) >> 3) << 3);
  
  int j_start = (jj-tile+1);
  int j_last   = (jj+1);
  int j_last_aligned = j_start + (((j_last-j_start) >> 3) << 3);

  int tile2 = tile+2; 
  //int kp1, kp5;
  int i2end = i_start % tile;
  int i_size = (i_start-i_last);
  int j_size = (j_last_aligned-j_start);
  int i_mat, j_mat;
  __m256i v_dml[i_size * j_size];

  for (i = i_start, i_mat = 0; i > i_last; i -= 1, i_mat += 1) {
    for (j = j_start, j_mat = 0; j < j_last_aligned; j += 8, j_mat += 1) {
      v_dml[i_mat + j_mat * i_size] = _mm256_loadu_si256((__m256i*) &ha_dmli[i%tile2][j]);
    }
  }

  for (k = k_start; k < k_last_aligned; k+= 8) {
    int kp1 = k+1, kp2 = k+2, kp3 = k+3, kp4 = k+4, kp5 = k+5, kp6 = k+6, kp7 = k+7;
    #pragma omp parallel for collapse(2)
    for (j = j_start, j_mat=0; j < j_last_aligned; j += 8, j_mat += i_size) {
  
      __m256 v_fm_k1j0 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+0]+kp1])), _mm_loadu_si128((__m128i*) &fm[indx[j+4]+kp1]), 1));
      __m256 v_fm_k1j1 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+1]+kp1])), _mm_loadu_si128((__m128i*) &fm[indx[j+5]+kp1]), 1));
      __m256 v_fm_k1j2 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+2]+kp1])), _mm_loadu_si128((__m128i*) &fm[indx[j+6]+kp1]), 1));
      __m256 v_fm_k1j3 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+3]+kp1])), _mm_loadu_si128((__m128i*) &fm[indx[j+7]+kp1]), 1));
      __m256 v_fm_k1j4 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+0]+kp5])), _mm_loadu_si128((__m128i*) &fm[indx[j+4]+kp5]), 1));
      __m256 v_fm_k1j5 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+1]+kp5])), _mm_loadu_si128((__m128i*) &fm[indx[j+5]+kp5]), 1));
      __m256 v_fm_k1j6 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+2]+kp5])), _mm_loadu_si128((__m128i*) &fm[indx[j+6]+kp5]), 1));
      __m256 v_fm_k1j7 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*) &fm[indx[j+3]+kp5])), _mm_loadu_si128((__m128i*) &fm[indx[j+7]+kp5]), 1));

      {
        __m256 _tmp0, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7, _tmp8;
        PERFORM_TRANSPOSE_AVX2(v_fm_k1j0, v_fm_k1j1, v_fm_k1j2, v_fm_k1j3, v_fm_k1j4, v_fm_k1j5, v_fm_k1j6, v_fm_k1j7)
      }

      __m256i v_add;   
      for (i = i2end, i_mat=j_mat; i >= 0; i -= 1, i_mat+=1) { 

          int *fmi = ha_fmi[i];

	  __m256i v_fmi_k0 = _mm256_set1_epi32(fmi[k  ]);
	  __m256i v_fmi_k1 = _mm256_set1_epi32(fmi[kp1]);
	  __m256i v_fmi_k2 = _mm256_set1_epi32(fmi[kp2]);
	  __m256i v_fmi_k3 = _mm256_set1_epi32(fmi[kp3]);
	  __m256i v_fmi_k4 = _mm256_set1_epi32(fmi[kp4]);
	  __m256i v_fmi_k5 = _mm256_set1_epi32(fmi[kp5]);
	  __m256i v_fmi_k6 = _mm256_set1_epi32(fmi[kp6]);
	  __m256i v_fmi_k7 = _mm256_set1_epi32(fmi[kp7]);
	
	  PERFORM_AVX2(v_fmi_k0, v_fm_k1j0, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k1, v_fm_k1j1, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k2, v_fm_k1j2, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k3, v_fm_k1j3, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k4, v_fm_k1j4, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k5, v_fm_k1j5, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k6, v_fm_k1j6, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k7, v_fm_k1j7, v_add, v_dml[i_mat]);

      }

      for (i=tile-1 ; i > i2end; i -= 1, i_mat+=1) {  

          int *fmi = ha_fmi[i];

	  __m256i v_fmi_k0 = _mm256_set1_epi32(fmi[k  ]);
	  __m256i v_fmi_k1 = _mm256_set1_epi32(fmi[kp1]);
	  __m256i v_fmi_k2 = _mm256_set1_epi32(fmi[kp2]);
	  __m256i v_fmi_k3 = _mm256_set1_epi32(fmi[kp3]);
	  __m256i v_fmi_k4 = _mm256_set1_epi32(fmi[kp4]);
	  __m256i v_fmi_k5 = _mm256_set1_epi32(fmi[kp5]);
	  __m256i v_fmi_k6 = _mm256_set1_epi32(fmi[kp6]);
	  __m256i v_fmi_k7 = _mm256_set1_epi32(fmi[kp7]);
	
	  PERFORM_AVX2(v_fmi_k0, v_fm_k1j0, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k1, v_fm_k1j1, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k2, v_fm_k1j2, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k3, v_fm_k1j3, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k4, v_fm_k1j4, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k5, v_fm_k1j5, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k6, v_fm_k1j6, v_add, v_dml[i_mat]);
	  PERFORM_AVX2(v_fmi_k7, v_fm_k1j7, v_add, v_dml[i_mat]);

      }


    } 
  }

  for (i = i_start, i_mat = 0; i > i_last; i -= 1, i_mat += 1) {
    for (j = j_start, j_mat = 0; j < j_last_aligned; j += 8, j_mat += 1) {
      _mm256_storeu_si256((__m256i*) &ha_dmli[i%tile2][j], v_dml[i_mat + j_mat * i_size]);
    }
  }




  for (i = i_start; i > i_last; i -= 1) {
      int *fmi = ha_fmi[i%tile];
      int *dmli= ha_dmli[i%tile2];
      for (k = k_start; k < k_last_aligned; k+= 8) {
        for (kin = 0; kin < 8; kin += 1) {
	  for ( ; j < j_last; j += 1) {
              k1j   = indx[j] + k+kin + 1;
              if ((fmi[k+kin] != INF) && (fm[k1j] != INF)) {
                  dmli[j] = MIN2(dmli[j], (fmi[k+kin] + fm[k1j]) );
              }
          }
        }
      }

      for ( ; k < k_last; k+= 1) {
          for (j = j_start; j < j_last_aligned; j += 8) {
            for (jin = 0; jin < 8; jin += 1) {
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

