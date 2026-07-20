import "slangpy";

extern static const uint TILE_SIZE = 16;
static const uint TILE_SIZE_SQRD = TILE_SIZE * TILE_SIZE;

struct Bounds {
    float2 min;
    float2 max;

    __init(float2 center, float2 halfSize) {
        min = center - halfSize;
        max = center + halfSize;
    }

    __init(int x, int y, int size) {
        min = float2(x, y) * size;
        max = float2(x + 1, y + 1) * size;
    }

    bool doesIntersect(Bounds oth) {
        const bool xIntersect = (min.x <= oth.max.x) && (max.x >= oth.min.x);
        const bool yIntersect = (min.y <= oth.max.y) && (max.y >= oth.min.y);
        return xIntersect && yIntersect;
    }
}

// works only for positive depth (could be made to work with negative depths but we cull those)
int64_t createKey(int tileId, float depth) {
    return ((int64_t)tileId << 32) | (int64_t)reinterpret<int>(depth);
}

int extractTileId(int64_t key) {
    return (uint)(key >> 32);
}

int getTileId(int x, int y, int2 tilesXY) {
    return x + y * tilesXY.x;
}

[Differentiable]
float2 cameraToScreen(float3 cameraPos, float4 intrinsics) {
    let focalX = intrinsics.x;
    let focalY = intrinsics.y;
    let centerX = intrinsics.z;
    let centerY = intrinsics.w;
    return float2(
        focalX * cameraPos.x / cameraPos.z + centerX,
        focalY * cameraPos.y / cameraPos.z + centerY
    );
}

[Differentiable]
float3x3 quaternionToMatrix(float4 quaternion) {
    let w = quaternion.x;
    let x = quaternion.y;
    let y = quaternion.z;
    let z = quaternion.w;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    return float3x3(
        1-2*(y2+z2), 2*(xy-z*w), 2*(xz+y*w),
        2*(xy+z*w), 1-2*(x2+z2), 2*(yz-x*w),
        2*(xz-y*w), 2*(yz+x*w), 1-2*(x2+y2)
    );
}

[Differentiable]
float2x3 getProjectionApproximation(float3 cameraPos, float2 focals) {
    let z2 = cameraPos.z * cameraPos.z;
    return float2x3(
        focals.x / cameraPos.z, 0, -focals.x * cameraPos.x / z2,
        0, focals.y / cameraPos.z, -focals.y * cameraPos.y / z2
    );
}

[Differentiable]
float2x2 compute2dCovariance(float3 scale, float4 quaternion, float3x3 viewTransform, float2x3 projApprox) {
    let rotationMat = quaternionToMatrix(normalize(quaternion));
    let scaleMatSqrd = float3x3(
        scale.x * scale.x, 0., 0.,
        0., scale.y * scale.y, 0.,
        0., 0., scale.z * scale.z
    );
    let sigma = mul(rotationMat, mul(scaleMatSqrd, transpose(rotationMat)));
    let jw = mul(projApprox, viewTransform);
    return mul(jw, mul(sigma, transpose(jw)));
}

int countIntersections(float2 screenPos, float2 variances, int2 tilesXY) {
    let gaussianBounds = Bounds(screenPos, 3 * sqrt(variances)); // roughly 99% confidence bounds
    
    int intersectionNum = 0;
    for (int j = 0; j < tilesXY.y; j++) {
        for (int i = 0; i < tilesXY.x; i++) {
            let tileBounds = Bounds(i, j, TILE_SIZE);
            if (gaussianBounds.doesIntersect(tileBounds)) intersectionNum++;
        }
    }
    
    return intersectionNum;
}

float3 getCameraCenter(float3x4 extrinsics) {
    let rotation = (float3x3)extrinsics;
    let translation = transpose(extrinsics)[3];
    return -mul(translation, rotation);
}

namespace ShNorm {
    static const float N0 = 0.28209479177387814;
    
    static const float N1 = 0.4886025119029199;
    
    static const float N2_m2 = 1.0925484305920792;
    static const float N2_m1 = 1.0925484305920792;
    static const float N2_0 = 0.31539156525252005;
    static const float N2_p1 = 1.0925484305920792;
    static const float N2_p2 = 0.5462742152960396;
    
    static const float N3_m3 = 0.5900435899266435;
    static const float N3_m2 = 2.890611442640554;
    static const float N3_m1 = 0.4570457994644658;
    static const float N3_0 = 0.3731763325901154;
    static const float N3_p1 = 0.4570457994644658;
    static const float N3_p2 = 1.445305721320277;
    static const float N3_p3 = 0.5900435899266435;
}

[Differentiable]
float3 computeColor(float3 direction, IDiffTensor<float, 3> shCoefs, uint idx, int maxDeg) {
    let x = direction.x;
    let y = direction.y;
    let z = direction.z;
    let xy = x * y;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    float3 color = 0;

    color += float3(shCoefs[0, idx, 0], shCoefs[0, idx, 1], shCoefs[0, idx, 2]) * ShNorm::N0;
    if (maxDeg < 1) return color;
    color += float3(shCoefs[1, idx, 0], shCoefs[1, idx, 1], shCoefs[1, idx, 2]) * ShNorm::N1 * y;
    color += float3(shCoefs[2, idx, 0], shCoefs[2, idx, 1], shCoefs[2, idx, 2]) * ShNorm::N1 * z;
    color += float3(shCoefs[3, idx, 0], shCoefs[3, idx, 1], shCoefs[3, idx, 2]) * ShNorm::N1 * x;
    if (maxDeg < 2) return color;
    color += float3(shCoefs[4, idx, 0], shCoefs[4, idx, 1], shCoefs[4, idx, 2]) * ShNorm::N2_m2 * xy;
    color += float3(shCoefs[5, idx, 0], shCoefs[5, idx, 1], shCoefs[5, idx, 2]) * ShNorm::N2_m1 * y*z;
    color += float3(shCoefs[6, idx, 0], shCoefs[6, idx, 1], shCoefs[6, idx, 2]) * ShNorm::N2_0 * (3*z2 - 1);
    color += float3(shCoefs[7, idx, 0], shCoefs[7, idx, 1], shCoefs[7, idx, 2]) * ShNorm::N2_p1 * x*z;
    color += float3(shCoefs[8, idx, 0], shCoefs[8, idx, 1], shCoefs[8, idx, 2]) * ShNorm::N2_p2 * (x2 - y2);
    if (maxDeg < 3) return color;
    color += float3(shCoefs[9, idx, 0], shCoefs[9, idx, 1], shCoefs[9, idx, 2]) * ShNorm::N3_m3 * y*(3*x2-y2);
    color += float3(shCoefs[10, idx, 0], shCoefs[10, idx, 1], shCoefs[10, idx, 2]) * ShNorm::N3_m2 * xy*z;
    color += float3(shCoefs[11, idx, 0], shCoefs[11, idx, 1], shCoefs[11, idx, 2]) * ShNorm::N3_m1 * y*(5*z2-1);
    color += float3(shCoefs[12, idx, 0], shCoefs[12, idx, 1], shCoefs[12, idx, 2]) * ShNorm::N3_0 * (5*z2*z-3*z);
    color += float3(shCoefs[13, idx, 0], shCoefs[13, idx, 1], shCoefs[13, idx, 2]) * ShNorm::N3_p1 * x*(5*z2-1);
    color += float3(shCoefs[14, idx, 0], shCoefs[14, idx, 1], shCoefs[14, idx, 2]) * ShNorm::N3_p2 * z*(x2-y2);
    color += float3(shCoefs[15, idx, 0], shCoefs[15, idx, 1], shCoefs[15, idx, 2]) * ShNorm::N3_p3 * x*(x2-3*y2);

    return color;
}

// sphere-frustrum test
bool isCulled(float3 cameraPos, float3 scales, float4 halfFovSinCos, float2 nearFar) {
    if (cameraPos.z < nearFar.x || cameraPos.z > nearFar.y) return true;
    
    let radius = 3. * max(max(scales.x, scales.y), scales.z); // 99+% confidence bounding sphere
    
    let sinx = halfFovSinCos.x;
    let cosx = halfFovSinCos.y;
    let leftOutNormal = -float3(cosx, 0., sinx);
    let rightOutNormal = -float3(-cosx, 0., sinx);
    if (dot(cameraPos, leftOutNormal) > radius ||
        dot(cameraPos, rightOutNormal) > radius) return true;
    
    let siny = halfFovSinCos.z;
    let cosy = halfFovSinCos.w;
    let upOutNormal = -float3(0., cosy, siny);
    let downOutNormal = -float3(0., -cosy, siny);
    if (dot(cameraPos, upOutNormal) > radius ||
        dot(cameraPos, downOutNormal) > radius) return true;

    return false;
}

[Differentiable]
void projectGaussians(
    uint idx,
    IDiffTensor<float, 2> worldPositions,
    IDiffTensor<float, 2> rotations,
    IDiffTensor<float, 2> scales,
    IDiffTensor<float, 3> shCoefficients,
    int2 tilesXY,
    no_diff float4 intrinsics,
    no_diff float4 halfFovSinCos,
    no_diff float3x4 extrinsics,
    no_diff float2 nearFar,
    bool useExp,
    no_diff float colorBias,
    int maxShDegree,
    IWDiffTensor<float, 2> screenPositions,
    IWTensor<float, 1> depths,
    IWDiffTensor<float, 2> covariances,
    IWDiffTensor<float, 2> colors,
    IWTensor<int, 1> counts)
{
    if (idx >= worldPositions.shape[0]) return;

    let worldPos = float3(worldPositions[idx, 0], worldPositions[idx, 1], worldPositions[idx, 2]);
    let cameraPos = mul(extrinsics, float4(worldPos, 1.));
    var scale = float3(scales[idx, 0], scales[idx, 1], scales[idx, 2]);
    if (useExp) scale = exp(scale);
    if (isCulled(cameraPos, scale, halfFovSinCos, nearFar)) {
        counts[idx] = 0;
        return;
    }

    let screenPos = cameraToScreen(cameraPos, intrinsics);
    let rotation = float4(rotations[idx, 0], rotations[idx, 1], rotations[idx, 2], rotations[idx, 3]);
    let covarianceMat = compute2dCovariance(scale, rotation, (float3x3)extrinsics, getProjectionApproximation(cameraPos, intrinsics.xy));
    let intersectionNum = countIntersections(screenPos, covarianceMat._11_22, tilesXY);
    if (intersectionNum == 0) {
        counts[idx] = 0;
        return;
    }
    
    let direction = normalize(no_diff(getCameraCenter(extrinsics)) - worldPos);
    let color = computeColor(direction, shCoefficients, idx, maxShDegree) + colorBias;
    screenPositions[idx, 0] = screenPos.x;
    screenPositions[idx, 1] = screenPos.y;
    depths[idx] = cameraPos.z;
    covariances[idx, 0] = covarianceMat._11;
    covariances[idx, 1] = covarianceMat._22;
    covariances[idx, 2] = covarianceMat._12;
    colors[idx, 0] = color.x;
    colors[idx, 1] = color.y;
    colors[idx, 2] = color.z;
    counts[idx] = intersectionNum;
}

void createGaussianInstances(
    uint idx,
    ITensor<int, 1> counts,
    ITensor<int, 1> offsets,
    ITensor<float, 2> positions,
    ITensor<float, 1> depths,
    ITensor<float, 2> covariances,
    int2 tilesXY,
    IWTensor<int, 1> instances,
    IWTensor<int64_t, 1> keys)
{
    if (idx >= counts.shape[0]) return;
    let count = counts[idx];
    if (count == 0) return;

    let offset = offsets[idx];
    for (int i = 0; i < count; i++) {
        instances[offset + i] = idx;
    }

    let position = float2(positions[idx, 0], positions[idx, 1]);
    let scales = float2(covariances[idx, 0], covariances[idx, 1]);
    let gaussianBounds = Bounds(position, 3 * sqrt(scales)); // roughly 99% confidence bounds
    let depth = depths[idx];
    
    int currInstance = 0;
    for (int j = 0; j < tilesXY.y; j++) {
        for (int i = 0; i < tilesXY.x; i++) {
            let tileBounds = Bounds(i, j, TILE_SIZE);
            if (!gaussianBounds.doesIntersect(tileBounds)) continue;
            keys[offset + currInstance] = createKey(getTileId(i, j, tilesXY), depth);
            currInstance++;
        }
    }
}

void findTileRangesNoShuffle(uint idx, ITensor<int64_t, 1> keys, IWTensor<int, 2> ranges) {
    let lastIdx = keys.shape[0] - 2;
    if (idx > lastIdx) return;
    
    let nextIdx = idx + 1;
    let prevTileId = extractTileId(keys[idx]);
    let nextTileId = extractTileId(keys[nextIdx]);
    if (idx == 0) ranges[prevTileId, 0] = 0;
    if (idx == lastIdx) ranges[nextTileId, 1] = keys.shape[0];
    if (prevTileId == nextTileId) return;
    ranges[prevTileId, 1] = nextIdx;
    ranges[nextTileId, 0] = nextIdx;
}

void findTileRanges(uint idx, ITensor<int64_t, 1> keys, IWTensor<int, 2> ranges) {
    let lastIdx = keys.shape[0] - 1;
    if (idx > lastIdx) return;

    let tileId = extractTileId(keys[idx]);
    var prevTileId = WaveReadLaneAt(tileId, WaveGetLaneIndex() - 1);
    if (WaveIsFirstLane()) {
        prevTileId = (idx == 0) ? tileId : extractTileId(keys[idx - 1]);
    }

    if (idx == 0) ranges[tileId, 0] = 0;
    if (idx == lastIdx) ranges[tileId, 1] = keys.shape[0];
    if (prevTileId == tileId) return;
    ranges[prevTileId, 1] = idx;
    ranges[tileId, 0] = idx;
}

struct Gaussians2D {
    float meanXs[TILE_SIZE_SQRD];
    float meanYs[TILE_SIZE_SQRD];
    float varXs[TILE_SIZE_SQRD];
    float varYs[TILE_SIZE_SQRD];
    float covars[TILE_SIZE_SQRD];
    float reds[TILE_SIZE_SQRD];
    float greens[TILE_SIZE_SQRD];
    float blues[TILE_SIZE_SQRD];
    float opacities[TILE_SIZE_SQRD];
}

groupshared Gaussians2D sharedGaussians;
groupshared Atomic<uint> finishedThreadNum;

uint2 getPixelCoords(uint localThreadId, uint tileId, int tilesAlongX) {
    let tileX = tileId % tilesAlongX;
    let tileY = tileId / tilesAlongX;
    let localPixelX = localThreadId % TILE_SIZE;
    let localPixelY = localThreadId / TILE_SIZE;
    return uint2(tileX * TILE_SIZE + localPixelX, tileY * TILE_SIZE + localPixelY);
}

T sigmoid<T>(T x) where T: __BuiltinFloatingPointType {
    return T(1.) / (T(1.) + exp(-x));
}

void loadGaussianIntoSharedMemory(int gaussianIdx, int storeIdx, IDiffTensor<float, 2> means, IDiffTensor<float, 2> covariances, IDiffTensor<float, 2> colors, IDiffTensor<float, 1> opacities) {
    let gaussianMean = float2(means[gaussianIdx, 0], means[gaussianIdx, 1]);
    let gaussianCovariance = float3(covariances[gaussianIdx, 0], covariances[gaussianIdx, 1], covariances[gaussianIdx, 2]);
    let gaussianColor = float3(colors[gaussianIdx, 0], colors[gaussianIdx, 1], colors[gaussianIdx, 2]);
    
    sharedGaussians.meanXs[storeIdx] = gaussianMean.x;
    sharedGaussians.meanYs[storeIdx] = gaussianMean.y;
    sharedGaussians.varXs[storeIdx] = gaussianCovariance.x;
    sharedGaussians.varYs[storeIdx] = gaussianCovariance.y;
    sharedGaussians.covars[storeIdx] = gaussianCovariance.z;
    sharedGaussians.reds[storeIdx] = gaussianColor.x;
    sharedGaussians.greens[storeIdx] = gaussianColor.y;
    sharedGaussians.blues[storeIdx] = gaussianColor.z;
    sharedGaussians.opacities[storeIdx] = opacities[gaussianIdx];
}

float computeGaussianAlpha(int sharedGaussianIdx, float2 pixelCenter, bool useSig) {
    let mean = float2(sharedGaussians.meanXs[sharedGaussianIdx], sharedGaussians.meanYs[sharedGaussianIdx]);
    let delta = pixelCenter - mean;

    let a = sharedGaussians.varXs[sharedGaussianIdx];
    let b = sharedGaussians.varYs[sharedGaussianIdx];
    let c = sharedGaussians.covars[sharedGaussianIdx];
    let invCovariance = float2x2(b, -c, -c, a) / (a * b - c * c);
    
    let storedOpacity = sharedGaussians.opacities[sharedGaussianIdx];
    let opacity = useSig ? sigmoid(storedOpacity) : storedOpacity;

    return opacity * exp(-.5 * dot(delta, mul(invCovariance, delta)));
}

// IMPORTANT slangpy always uses 1d indices for the threads, even when specifying a 2D grid
[BackwardDerivative(renderGaussiansBwd)]
void renderGaussians(
    uint2, uint threadId,
    int2 tilesXY,
    ITensor<int, 1> instances,
    ITensor<int, 2> tileRanges,
    DiffTensor<float, 2> means,
    DiffTensor<float, 2> covariances,
    DiffTensor<float, 2> colors,
    DiffTensor<float, 1> opacities,
    bool useSig,
    no_diff float alphaThres,
    no_diff float maxAlpha,
    no_diff float minTransmittance,
    bool saveBackpropData,
    WDiffTensor<float, 3> image,
    IRWTensor<float, 2> accumulatedTransmittances,
    IRWTensor<int, 2> processedGaussianCount)
{
    let workgroupSize = WorkgroupSize().x;
    let tileId = threadId / workgroupSize; // here we assume that each tile is mapped to a workgroup
    let localThreadId = threadId % workgroupSize;
    let pixelCoords = getPixelCoords(localThreadId, tileId, tilesXY.x);
    let pixelCenter = pixelCoords + .5;
    let isValidPixel = (pixelCoords.x < image.shape[1]) && (pixelCoords.y < image.shape[0]); // IMPORTANT the image tensor shape should be (H,W,C) NOT (C,H,W)
    let tileRange = int2(tileRanges[tileId, 0], tileRanges[tileId, 1]);
    let maxGaussians = tileRange.y - tileRange.x;
    
    var color = float3(.0);
    var transmittance = 1.;
    var processedGaussianNum = 0;
    if (maxGaussians > 0) { // if no gaussians were assigned to this tile/workgroup, can safely skip the alpha blending and synchronization processes
        if (localThreadId == 0) finishedThreadNum.store(0);
        GroupMemoryBarrierWithGroupSync();
        if (!isValidPixel) finishedThreadNum.increment();

        let storeIdx = localThreadId;
        var loadIdx = tileRange.x + localThreadId;
        var hasFinished = !isValidPixel;
        do {
            if (loadIdx < tileRange.y) {
                loadGaussianIntoSharedMemory(instances[loadIdx], storeIdx, means, covariances, colors, opacities);
            }
            GroupMemoryBarrierWithGroupSync();

            if (!hasFinished) {
                for (int i = 0; i < workgroupSize; i++, processedGaussianNum++) {
                    if (processedGaussianNum == maxGaussians) {
                        hasFinished = true;
                        finishedThreadNum.increment();
                        break;
                    }
                    let alpha = min(computeGaussianAlpha(i, pixelCenter, useSig), maxAlpha);
                    if (alpha <= alphaThres) continue;

                    let nextTransmittance = transmittance * (1 - alpha);
                    if (nextTransmittance <= minTransmittance) {
                        hasFinished = true;
                        finishedThreadNum.increment();
                        break;
                    }
                    let gaussianColor = float3(sharedGaussians.reds[i], sharedGaussians.greens[i], sharedGaussians.blues[i]);
                    color += gaussianColor * alpha * transmittance;
                    transmittance = nextTransmittance;
                }
            }
            loadIdx += workgroupSize;

            GroupMemoryBarrierWithGroupSync();
        } while (finishedThreadNum.load() != workgroupSize);
    }

    if (!isValidPixel) return;
    image[pixelCoords.y, pixelCoords.x, 0] = color.x;
    image[pixelCoords.y, pixelCoords.x, 1] = color.y;
    image[pixelCoords.y, pixelCoords.x, 2] = color.z;
    if (!saveBackpropData) return;
    accumulatedTransmittances[pixelCoords.y, pixelCoords.x] = transmittance;
    processedGaussianCount[pixelCoords.y, pixelCoords.x] = processedGaussianNum;
}

groupshared Atomic<int> sharedBlockMaxGaussians;

int getBlockMaxGaussians(uint localThreadId, int threadMaxGaussians) {
    if (localThreadId == 0) sharedBlockMaxGaussians.store(0);
    GroupMemoryBarrierWithGroupSync();

    let waveMaxGaussians = WaveActiveMax(threadMaxGaussians);
    if (WaveIsFirstLane()) sharedBlockMaxGaussians.reduceMax(waveMaxGaussians);
    GroupMemoryBarrierWithGroupSync();

    return sharedBlockMaxGaussians.load();
}

void renderGaussiansBwd(
    uint2, uint threadId,
    int2 tilesXY,
    ITensor<int, 1> instances,
    ITensor<int, 2> tileRanges,
    DiffTensor<float, 2> means,
    DiffTensor<float, 2> covariances,
    DiffTensor<float, 2> colors,
    DiffTensor<float, 1> opacities,
    bool useSig,
    float alphaThres,
    float maxAlpha,
    float, bool,
    WDiffTensor<float, 3> image,
    IRWTensor<float, 2> accumulatedTransmittances,
    IRWTensor<int, 2> processedGaussianCount)
{
    let workgroupSize = WorkgroupSize().x;
    let tileId = threadId / workgroupSize;
    let localThreadId = threadId % workgroupSize;
    let pixelCoords = getPixelCoords(localThreadId, tileId, tilesXY.x);
    let pixelCenter = pixelCoords + .5;
    let isValidPixel = (pixelCoords.x < image.shape[1]) && (pixelCoords.y < image.shape[0]);
    let tileRange = int2(tileRanges[tileId, 0], tileRanges[tileId, 1]);
    if (tileRange.y <= tileRange.x) return; // no gaussians assigned were assigned to the tile/block, can return

    let threadMaxGaussians = isValidPixel ? processedGaussianCount[pixelCoords.y, pixelCoords.x] : 0;
    let blockMaxGaussians = getBlockMaxGaussians(localThreadId, threadMaxGaussians);
    if (blockMaxGaussians <= 0) return; // no thread in the tile/block has processed any gaussians, can return
    
    var loadIdx = tileRange.x + blockMaxGaussians - 1 - localThreadId;
    let storeIdx = localThreadId;
    var batchMaxGaussianInstance = tileRange.x + blockMaxGaussians - 1;
    var batchMinGaussianInstance = batchMaxGaussianInstance - workgroupSize + 1;
    let threadMaxGaussianInstance = tileRange.x + threadMaxGaussians - 1;
    var currGaussianInstance = threadMaxGaussianInstance;
    // xGrad is to be thought as the derivate of the loss wrt x
    let pixelGrad = !isValidPixel ? float3(0.) : float3(
        image._grad_in[pixelCoords.y, pixelCoords.x, 0],
        image._grad_in[pixelCoords.y, pixelCoords.x, 1],
        image._grad_in[pixelCoords.y, pixelCoords.x, 2]
    );
    float accumTransm = isValidPixel ? accumulatedTransmittances[pixelCoords.y, pixelCoords.x] : 1.;
    var backSum = float3(.0);
    while (batchMaxGaussianInstance >= tileRange.x) {
        if (loadIdx >= tileRange.x) {
            loadGaussianIntoSharedMemory(instances[loadIdx], storeIdx, means, covariances, colors, opacities);
        }
        GroupMemoryBarrierWithGroupSync();

        if (isValidPixel && batchMinGaussianInstance <= threadMaxGaussianInstance) {
            for (int i = max(0, batchMaxGaussianInstance - currGaussianInstance);
                i < workgroupSize && currGaussianInstance >= tileRange.x;
                i++, currGaussianInstance--)
            {
                let mean = float2(sharedGaussians.meanXs[i], sharedGaussians.meanYs[i]);
                let a = sharedGaussians.varXs[i];
                let b = sharedGaussians.varYs[i];
                let c = sharedGaussians.covars[i];
                let invSigma = float2x2(b, -c, -c, a) / (a * b - c * c);
                let storedOpacity = sharedGaussians.opacities[i];
                let opacity = useSig ? sigmoid(storedOpacity) : storedOpacity;
                
                let delta = mean - pixelCenter;
                let invSigmaDelta = mul(invSigma, delta);
                let eSig = exp(-.5 * dot(delta, invSigmaDelta));
                let alpha = min(opacity * eSig, maxAlpha);
                if (alpha <= alphaThres) continue;
                
                accumTransm /= (1. - alpha);
                let dPixeldColor = float3(alpha * accumTransm);
                let color = float3(sharedGaussians.reds[i], sharedGaussians.greens[i], sharedGaussians.blues[i]);
                let dPixeldAlpha = color * accumTransm - backSum / (1 - alpha);
                let dAlphadOpacity = eSig;
                let dAlphadSig = -alpha;
                let dSigdMean = invSigmaDelta;
                let dSigdSigma = -.5 * float3(
                    invSigmaDelta.x * invSigmaDelta.x,
                    invSigmaDelta.y * invSigmaDelta.y,
                    2 * invSigmaDelta.x * invSigmaDelta.y
                );
                
                let colorGrad = dPixeldColor * pixelGrad;
                let alphaGrad = dot(dPixeldAlpha, pixelGrad);
                let opacityGrad = dAlphadOpacity * alphaGrad;
                let sigGrad = dAlphadSig * alphaGrad;
                let meanGrad = dSigdMean * sigGrad;
                let sigmaGrad = dSigdSigma * sigGrad;
                
                let gaussianIdx = instances[currGaussianInstance];
                means._grad_out[gaussianIdx, 0] += meanGrad.x;
                means._grad_out[gaussianIdx, 1] += meanGrad.y;
                covariances._grad_out[gaussianIdx, 0] += sigmaGrad.x;
                covariances._grad_out[gaussianIdx, 1] += sigmaGrad.y;
                covariances._grad_out[gaussianIdx, 2] += sigmaGrad.z;
                colors._grad_out[gaussianIdx, 0] += colorGrad.x;
                colors._grad_out[gaussianIdx, 1] += colorGrad.y;
                colors._grad_out[gaussianIdx, 2] += colorGrad.z;
                opacities._grad_out[gaussianIdx] += opacityGrad;
                
                backSum += color * alpha * accumTransm;
            }
        }

        batchMaxGaussianInstance -= workgroupSize;
        batchMinGaussianInstance -= workgroupSize;
        loadIdx -= workgroupSize;
    }
}
