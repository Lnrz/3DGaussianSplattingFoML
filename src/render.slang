import "slangpy";

struct Bounds {
    float2 min;
    float2 max;

    __init(float2 center, float2 halfSize) {
        min = center - halfSize;
        max = center + halfSize;
    }

    __init(int x, int y, int tileSize) {
        min = float2(x, y) * tileSize;
        max = float2(x + 1, y + 1) * tileSize;
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

float2x3 getProjectionApproximation(float3 cameraPos, float2 focals) {
    let z2 = cameraPos.z * cameraPos.z;
    return float2x3(
        focals.x / cameraPos.z, 0, -focals.x * cameraPos.x / z2,
        0, focals.y / cameraPos.z, -focals.y * cameraPos.y / z2
    );
}

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
            let tileBounds = Bounds(i, j, 16); // MAGIC NUMBER ALERT
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

float3 computeColor(float3 direction, ITensor<float, 3> shCoefs, uint idx, int maxDeg) {
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

void projectGaussians(
    uint idx,
    ITensor<float, 2> worldPositions,
    ITensor<float, 2> rotations,
    ITensor<float, 2> scales,
    ITensor<float, 3> shCoefficients,
    int2 tilesXY,
    float4 intrinsics,
    float4 halfFovSinCos,
    float3x4 extrinsics,
    float2 nearFar,
    bool useExp,
    float colorBias,
    int maxShDegree,
    IWTensor<float, 2> screenPositions,
    IWTensor<float, 1> depths,
    IWTensor<float, 2> covariances,
    IWTensor<float, 2> colors,
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
    
    let direction = normalize(getCameraCenter(extrinsics) - worldPos);
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
            let tileBounds = Bounds(i, j, 16);  // MAGIC NUMBER ALERT
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

void renderGaussians(
    uint2 idx,
    ITensor<int, 1> indices,
    ITensor<int, 1> tileRanges,
    ITensor<float, 2> means,
    ITensor<float, 2> covariances,
    ITensor<float, 2> colors,
    ITensor<float, 1> opacities,
    float alphaThres,
    float minTransmittance,
    bool saveFinalTransmittances,
    IWTensor<float, 3> image,
    IWTensor<float, 2> finalTransmittances) {
    return;
}
