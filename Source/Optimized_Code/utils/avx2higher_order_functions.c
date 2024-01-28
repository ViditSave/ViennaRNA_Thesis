#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ViennaRNA/utils/basic.h"
#include "ViennaRNA/utils/cpu.h"


typedef int (proto_fun_zip_reduce)(const int  *a,
                                   const int  *b,
                                   int        size);


/*
 #################################
 # PRIVATE FUNCTION DECLARATIONS #
 #################################
 */
static int
zip_add_min_dispatcher(const int  *a,
                       const int  *b,
                       int        size);


static int
fun_zip_add_min_default(const int *e1,
                        const int *e2,
                        int       count);

void
vrna_add_min_new_default(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli);


#if VRNA_WITH_SIMD_AVX512
int
vrna_fun_zip_add_min_avx512(const int *e1,
                            const int *e2,
                            int       count);

int64_t
vrna_add_min_new_avx2(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli,
		   int64_t flop_count);

#endif

#if VRNA_WITH_SIMD_SSE41
int
vrna_fun_zip_add_min_sse41(const int  *e1,
                           const int  *e2,
                           int        count);

int
vrna_add_min_new_sse41(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli,
		   int64_t flop_count);

#endif


static proto_fun_zip_reduce *fun_zip_add_min = &zip_add_min_dispatcher;


/*
 #################################
 # BEGIN OF FUNCTION DEFINITIONS #
 #################################
 */
PUBLIC void
vrna_fun_dispatch_disable(void)
{
  fun_zip_add_min = &fun_zip_add_min_default;
}


PUBLIC void
vrna_fun_dispatch_enable(void)
{
  fun_zip_add_min = &zip_add_min_dispatcher;
}


PUBLIC int
vrna_fun_zip_add_min(const int  *e1,
                     const int  *e2,
                     int  count)
{
  return (*fun_zip_add_min)(e1, e2, count);
}


#define MAX_NEW(i, j) (((i) > (j)) ? (i) : (j))


PUBLIC int64_t
vrna_add_min_new(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli,
		   int64_t flop_count)
{ 
  //vrna_add_min_new_default(fc, ii,jj,tile,ha_fmi, ha_dmli);
  return vrna_add_min_new_avx2(fc, ii,jj, tile, ha_fmi, ha_dmli, flop_count); 
}



/*
 #################################
 # STATIC helper functions below #
 #################################
 */

/* zip_add_min() dispatcher */
static int
zip_add_min_dispatcher(const int  *a,
                       const int  *b,
                       int        size)
{
  unsigned int features = vrna_cpu_simd_capabilities();

#if VRNA_WITH_SIMD_AVX512
  if (features & VRNA_CPU_SIMD_AVX512F) {
    fun_zip_add_min = &vrna_fun_zip_add_min_avx512;
    goto exec_fun_zip_add_min;
  }

#endif

#if VRNA_WITH_SIMD_SSE41
  if (features & VRNA_CPU_SIMD_SSE41) {
    fun_zip_add_min = &vrna_fun_zip_add_min_sse41;
    goto exec_fun_zip_add_min;
  }

#endif

  fun_zip_add_min = &fun_zip_add_min_default;

exec_fun_zip_add_min:

  return (*fun_zip_add_min)(a, b, size);
}


static int
fun_zip_add_min_default(const int *e1,
                        const int *e2,
                        int       count)
{
  int i;
  int decomp = INF;

  for (i = 0; i < count; i++) {
    if ((e1[i] != INF) && (e2[i] != INF)) {
      const int en = e1[i] + e2[i];
      decomp = MIN2(decomp, en);
    }
  }

  return decomp;
}



void
vrna_add_min_new_default(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli) {

  int                       i, j, k, k1j, decomp, *indx, *fm;
  int 			    k_mid, k1j_mid, k_end;
  const int section_up = ii - i + 1;
  indx          = fc->jindx;
  fm            = fc->matrices->fML;
 
  for (i = MAX_NEW(ii,0); i > MAX_NEW((ii-tile), 0); i -= 1) {
    int *fmi = ha_fmi[i%tile];
    int *dmli= ha_dmli[i%(tile+2)];
    for (k_mid = (ii + 2); k_mid < (jj - tile - 3); k_mid++) {
      for (j = (jj-tile+1); j < (jj+1); j += 1) {
        k1j_mid   = indx[j] + k_mid + 1;
        if ((fmi[k_mid] != INF) && (fm[k1j_mid] != INF)) {
          dmli[j] = MIN2(dmli[j], (fmi[k_mid] + fm[k1j_mid]) );
        }
      }
    }
  }


}
