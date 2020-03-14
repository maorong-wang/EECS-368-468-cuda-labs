#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* d_input,uint8_t* d_bin,uint32_t *t_bin);

/* Include below the function headers of any other functions that you implement */
void opt_2dhisto_init(uint32_t **input,uint32_t* (&d_input),uint8_t* (&d_bin),uint32_t* (&t_bin));
void opt_2dhisto_finalize(uint32_t* &d_input,uint8_t* &d_bin,uint32_t* &t_bin,uint8_t* kernel_bin);

#endif
