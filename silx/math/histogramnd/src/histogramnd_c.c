#include "histogramnd_c.h"

/*=====================
 * double sample
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T double
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#include "histogramnd_template.c"

/*=====================
 * float sample
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T float
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#include "histogramnd_template.c"

/*=====================
 * int32_t sample
 * =====================
*/
#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T double
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T float
#include "histogramnd_template.c"

#ifdef HISTO_SAMPLE_T
#undef HISTO_SAMPLE_T
#endif
#define HISTO_SAMPLE_T int32_t
#ifdef HISTO_WEIGHT_T
#undef HISTO_WEIGHT_T
#endif
#define HISTO_WEIGHT_T int32_t
#include "histogramnd_template.c"
