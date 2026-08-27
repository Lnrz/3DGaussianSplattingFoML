import slangpy;
import math;
import rand;


void accumulateNorms(uint idx, IRWTensor<float, 1> accum, ITensor<float, 2> gradients, float2 scales) {
    if (idx >= accum.shape[0]) return;

    let viewPosGrad = float2(gradients[idx, 0] , gradients[idx, 1]) * scales;
    accum[idx] += length(viewPosGrad);
}

void countCopies(uint idx,
    ITensor<float, 1> accumNorms,
    ITensor<int, 1> viewCounters,
    ITensor<float, 2> scales,
    ITensor<float, 1> opacities,
    ITensor<float, 1> gaussianImageRadii,
    float magnitudeThreshold,
    bool useExp,
    float worldRadiusThreshold,
    bool useSig,
    float opacityThreshold,
    float imageRadiusThreshold,
    IWTensor<int, 1> counts)
{
    if (idx >= scales.shape[0]) return;

    let opacity = useSig ? sigmoid(opacities[idx]) : opacities[idx];
    if (opacity < opacityThreshold) {
        counts[idx] = 0;
        return;
    }

    let viewCounter = viewCounters[idx];
    let avgNorm = (viewCounter != 0) ? accumNorms[idx] / viewCounter : 0.;
    if (avgNorm >= magnitudeThreshold) {
        counts[idx] = 2;
        return;
    }

    var scale = float3(scales[idx, 0], scales[idx, 1], scales[idx, 2]);
    if (useExp) scale = exp(scale);
    let maxWorldRadius = max(max(scale.x, scale.y), scale.z);
    counts[idx] = (maxWorldRadius > worldRadiusThreshold || gaussianImageRadii[idx] > imageRadiusThreshold)? 0 : 1;
}

void copyGaussian(uint srcIdx, int dstIdx,
    float3 mean, float4 rotation, float3 scale, float opacity, ITensor<float, 3> inShCoefs,
    IWTensor<float, 2> outMeans, IWTensor<float, 2> outRotations, IWTensor<float, 2> outScales, IWTensor<float, 3> outShCoefs, IWTensor<float, 1> outOpacities)
{
    for (int i = 0; i < 3; i++) {
        outMeans[dstIdx, i] = mean[i];
    }
    for (int i = 0; i < 4; i++) {
        outRotations[dstIdx, i] = rotation[i];
    }
    for (int i = 0; i < 3; i++) {
        outScales[dstIdx, i] = scale[i];
    }
    outOpacities[dstIdx] = opacity;
    for (int coef = 0; coef < 16; coef++) {
        for (int ch = 0; ch < 3; ch++) {
            outShCoefs[coef, dstIdx, ch] = inShCoefs[coef, srcIdx, ch];
        }
    }
}

Tuple<float3, float3> sampleGaussian(inout uint64_t state, float3 mean, float3 scale, float4 quaternion) {
    float us[6];
    for (int i = 0; i < us.getCount(); i++) {
        us[i] = pcg32(state);
    }

    float3 samples[2];
    for (int i = 0; i < 3; i++) {
        let ns = boxMuller(us[2 * i], us[2 * i + 1]);
        samples[0][i] = ns.x;
        samples[1][i] = ns.y;
    }

    let rotation = quaternionToMatrix(normalize(quaternion));
    for (int i = 0; i < samples.getCount(); i++) {
        samples[i] = mul(rotation, samples[i] * scale) + mean;
    }

    return makeTuple(samples[0], samples[1]);
}

void createCopies(uint idx, uint64_t seed,
    ITensor<int, 1> counts,
    ITensor<int, 1> cumCounts,
    ITensor<float, 2> inMeans,
    ITensor<float, 2> inRotations,
    ITensor<float, 2> inScales,
    ITensor<float, 3> inShCoefs,
    ITensor<float, 1> inOpacities,
    bool useExp,
    float scaleThreshold,
    float scaleFactor,
    bool resetOpacity,
    float opacityResetValue,
    IWTensor<float, 2> outMeans,
    IWTensor<float, 2> outRotations,
    IWTensor<float, 2> outScales,
    IWTensor<float, 3> outShCoefs,
    IWTensor<float, 1> outOpacities)
{
    let copiesNum = counts[idx];
    if (copiesNum == 0) return;
    let offset = cumCounts[idx] - copiesNum;

    let mean = float3(inMeans[idx, 0], inMeans[idx, 1], inMeans[idx, 2]);
    let rotation = float4(inRotations[idx, 0], inRotations[idx, 1], inRotations[idx, 2], inRotations[idx, 3]);
    let storedScale = float3(inScales[idx, 0], inScales[idx, 1], inScales[idx, 2]);
    let opacity = resetOpacity ? opacityResetValue : inOpacities[idx];
    if (copiesNum == 1) {
        copyGaussian(idx, offset, mean, rotation, storedScale, opacity, inShCoefs, outMeans, outRotations, outScales, outShCoefs, outOpacities);
        return;
    }

    let scale = useExp ? exp(storedScale) : storedScale;
    if (max(scale.x, max(scale.y, scale.z)) <= scaleThreshold) {
        copyGaussian(idx, offset, mean, rotation, storedScale, opacity, inShCoefs, outMeans, outRotations, outScales, outShCoefs, outOpacities);
        copyGaussian(idx, offset + 1, mean, rotation, storedScale, opacity, inShCoefs, outMeans, outRotations, outScales, outShCoefs, outOpacities);
    } else {
        let scaleToStore = useExp ? log(scale / scaleFactor) : scale / scaleFactor;

        uint64_t state = getThreadState(idx, seed);
        pcg32(state);
        let samples = sampleGaussian(state, mean, scale, rotation);

        copyGaussian(idx, offset, samples._0, rotation, scaleToStore, opacity, inShCoefs, outMeans, outRotations, outScales, outShCoefs, outOpacities);
        copyGaussian(idx, offset + 1, samples._1, rotation, scaleToStore, opacity, inShCoefs, outMeans, outRotations, outScales, outShCoefs, outOpacities);
    }
}
