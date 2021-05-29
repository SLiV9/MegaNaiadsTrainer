#pragma once

constexpr size_t NUM_SEATS = 4;
constexpr size_t NUM_STATE_SETS = 1 + NUM_SEATS + NUM_SEATS;
constexpr size_t NUM_VIEW_SETS = 1 + NUM_SEATS + 1;
constexpr size_t NUM_CARDS = 32;
constexpr size_t NUM_SUITS = 4;
constexpr size_t NUM_FACES_PER_SUIT = NUM_CARDS / NUM_SUITS;
constexpr size_t NUM_CARDS_PER_HAND = 3;

constexpr size_t NUM_PERSONALITIES = 4;
constexpr size_t NUM_BRAINS_PER_PERSONALITY = 25;

#define ENABLE_CUDA torch::cuda::is_available()
