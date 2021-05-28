#pragma once

#define NUM_STATE_SETS 6
#define NUM_VIEW_SETS 6
#define NUM_CARDS 32

#define NUM_PERSONALITIES 4
#define NUM_BRAINS_PER_PERSONALITY 25

#define ENABLE_CUDA torch::cuda::is_available()
