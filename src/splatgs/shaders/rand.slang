module rand;

import math;


// Get thread initial state using SplitMix64
public uint64_t getThreadState(uint64_t threadId, uint64_t seed) {
    uint64_t x = threadId ^ (seed + 0x9e3779b97f4a7c15ULL);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// PCG32
// Return a float in range (0-1]
// Taken from https://github.com/imneme/pcg-c-basic
public float pcg32(inout uint64_t state) {
    let oldState = state;

    state = oldState * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    let xorShifted = uint32_t(((oldState >> 18u) ^ oldState) >> 27u);
    let rot = uint32_t(oldState >> 59u);
    let resInt = (xorShifted >> rot) | (xorShifted << ((-rot) & 31));

    return 2. - reinterpret<float>(0x3F800000u | (resInt >> 9));
}

// Box-Muller Transform
// Transform u0 and u1 from U(0,1) to N(0,1)
public float2 boxMuller(float u0, float u1) {
    return float2(sin(TAU * u0), cos(TAU * u0)) * sqrt(-2. * log(u1));
}
