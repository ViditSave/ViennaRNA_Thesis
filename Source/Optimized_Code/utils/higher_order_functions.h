#ifndef VIENNA_RNA_PACKAGE_UTILS_FUN_H
#define VIENNA_RNA_PACKAGE_UTILS_FUN_H

void
vrna_fun_dispatch_disable(void);


void
vrna_fun_dispatch_enable(void);


int
vrna_fun_zip_add_min(const int  *e1,
                     const int  *e2,
                     int        count);

PUBLIC int64_t
vrna_add_min_new(vrna_fold_compound_t *fc,
                   int  ii,
                   int  jj,
                   int  tile,
  		   int  **ha_fmi,
                   int  **ha_dmli,
		   int64_t flop_count);


#endif
