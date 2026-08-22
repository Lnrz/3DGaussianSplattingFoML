import "slangpy";

extern static const uint TILE_SIZE = 16;
static const uint TILE_SIZE_SQRD = TILE_SIZE * TILE_SIZE;

// works only for positive depth (could be made to work with negative depths but they are culled)
uint64_t createKey(int tileId, float depth) {
    return ((uint64_t)reinterpret<uint>(tileId) << 32) | (uint64_t)reinterpret<uint>(depth);
}

uint extractTileId(int64_t key) {
    return (uint)(reinterpret<uint64_t>(key) >> 32);
}

int getTileId(int x, int y, int2 tilesXY) {
    return x + y * tilesXY.x;
}

[Differentiable]
float2 cameraToScreen(float3 cameraPos, no_diff float4 intrinsics) {
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
float2x3 getProjectionApproximation(float3 cameraPos, no_diff float2 focals) {
    let z2 = cameraPos.z * cameraPos.z;
    return float2x3(
        focals.x / cameraPos.z, 0, -focals.x * cameraPos.x / z2,
        0, focals.y / cameraPos.z, -focals.y * cameraPos.y / z2
    );
}

[Differentiable]
float2x2 compute2dCovariance(float3 scale, float4 quaternion, no_diff float3x3 viewTransform, float2x3 projApprox) {
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

bool isValidFloat<T, let n : int>(vector<T, n> x) where T : __BuiltinFloatingPointType {
    return all(!isnan(x) && !isinf(x));
}

int countIntersections(float2 screenPos, float3 covariances, int2 tilesXY) {
    let halfSize = 3. * sqrt(covariances.xy);
    let minPos = screenPos - halfSize;
    let maxPos = screenPos + halfSize;
    let minTileCoords = max(int2(0, 0), int2(floor(minPos / TILE_SIZE)));
    let maxTileCoords = min(tilesXY - 1, int2(floor(maxPos / TILE_SIZE)));
    if (any(minTileCoords > maxTileCoords)) return 0;

    let bboxSpan = maxTileCoords - minTileCoords + 1;
    return bboxSpan.x * bboxSpan.y;
}

float3 getCameraCenter(float3x4 extrinsics) {
    let rotation = (float3x3)extrinsics;
    let translation = extrinsics._14_24_34;
    return -mul(translation, rotation);
}

[Differentiable]
float3 getDirection(no_diff float3 from, float3 to) {
    return normalize(to - from);
}

namespace ShNorm {
    static const float N0 = 0.28209479177387814;
    
    static const float N1 = 0.4886025119029199;
    
    static const float N2_m2 = 1.0925484305920792;
    static const float N2_m1 = -1.0925484305920792;
    static const float N2_0 = 0.31539156525252005;
    static const float N2_p1 = -1.0925484305920792;
    static const float N2_p2 = 0.5462742152960396;
    
    static const float N3_m3 = -0.5900435899266435;
    static const float N3_m2 = 2.890611442640554;
    static const float N3_m1 = -0.4570457994644658;
    static const float N3_0 = 0.3731763325901154;
    static const float N3_p1 = -0.4570457994644658;
    static const float N3_p2 = 1.445305721320277;
    static const float N3_p3 = -0.5900435899266435;
}

[BackwardDerivative(computeColorBwd)]
float3 computeColor(float3 direction, DiffTensor<float, 3> shCoefs, uint idx, int maxDeg) {
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
    color -= float3(shCoefs[1, idx, 0], shCoefs[1, idx, 1], shCoefs[1, idx, 2]) * ShNorm::N1 * y;
    color += float3(shCoefs[2, idx, 0], shCoefs[2, idx, 1], shCoefs[2, idx, 2]) * ShNorm::N1 * z;
    color += float3(shCoefs[3, idx, 0], shCoefs[3, idx, 1], shCoefs[3, idx, 2]) * ShNorm::N1 * x;
    if (maxDeg < 2) return color;
    color += float3(shCoefs[4, idx, 0], shCoefs[4, idx, 1], shCoefs[4, idx, 2]) * ShNorm::N2_m2 * xy;
    color += float3(shCoefs[5, idx, 0], shCoefs[5, idx, 1], shCoefs[5, idx, 2]) * ShNorm::N2_m1 * y*z;
    color += float3(shCoefs[6, idx, 0], shCoefs[6, idx, 1], shCoefs[6, idx, 2]) * ShNorm::N2_0 * (2*z2-x2-y2);
    color += float3(shCoefs[7, idx, 0], shCoefs[7, idx, 1], shCoefs[7, idx, 2]) * ShNorm::N2_p1 * x*z;
    color += float3(shCoefs[8, idx, 0], shCoefs[8, idx, 1], shCoefs[8, idx, 2]) * ShNorm::N2_p2 * (x2-y2);
    if (maxDeg < 3) return color;
    color += float3(shCoefs[9, idx, 0], shCoefs[9, idx, 1], shCoefs[9, idx, 2]) * ShNorm::N3_m3 * y*(3*x2-y2);
    color += float3(shCoefs[10, idx, 0], shCoefs[10, idx, 1], shCoefs[10, idx, 2]) * ShNorm::N3_m2 * xy*z;
    color += float3(shCoefs[11, idx, 0], shCoefs[11, idx, 1], shCoefs[11, idx, 2]) * ShNorm::N3_m1 * y*(4*z2-x2-y2);
    color += float3(shCoefs[12, idx, 0], shCoefs[12, idx, 1], shCoefs[12, idx, 2]) * ShNorm::N3_0 * z*(2*z2-3*x2-3*y2);
    color += float3(shCoefs[13, idx, 0], shCoefs[13, idx, 1], shCoefs[13, idx, 2]) * ShNorm::N3_p1 * x*(4*z2-x2-y2);
    color += float3(shCoefs[14, idx, 0], shCoefs[14, idx, 1], shCoefs[14, idx, 2]) * ShNorm::N3_p2 * z*(x2-y2);
    color += float3(shCoefs[15, idx, 0], shCoefs[15, idx, 1], shCoefs[15, idx, 2]) * ShNorm::N3_p3 * x*(x2-3*y2);

    return color;
}

void computeColorBwd(inout DifferentialPair<float3> direction, DiffTensor<float, 3> shCoefs, uint idx, int maxDeg, float3 colorGrad) {
    let x = direction.p.x;
    let y = direction.p.y;
    let z = direction.p.z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;
    let xyz = xy * z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    var directionGrad = float3(0.);

    shCoefs._grad_out[0, idx, 0] += ShNorm::N0 * colorGrad[0];
    shCoefs._grad_out[0, idx, 1] += ShNorm::N0 * colorGrad[1];
    shCoefs._grad_out[0, idx, 2] += ShNorm::N0 * colorGrad[2];

    if (maxDeg < 1) {
        direction = diffPair(direction.p, directionGrad);
        return;
    }
    shCoefs._grad_out[1, idx, 0] -= ShNorm::N1 * y * colorGrad[0];
    shCoefs._grad_out[1, idx, 1] -= ShNorm::N1 * y * colorGrad[1];
    shCoefs._grad_out[1, idx, 2] -= ShNorm::N1 * y * colorGrad[2];

    shCoefs._grad_out[2, idx, 0] += ShNorm::N1 * z * colorGrad[0];
    shCoefs._grad_out[2, idx, 1] += ShNorm::N1 * z * colorGrad[1];
    shCoefs._grad_out[2, idx, 2] += ShNorm::N1 * z * colorGrad[2];
    
    shCoefs._grad_out[3, idx, 0] += ShNorm::N1 * x * colorGrad[0];
    shCoefs._grad_out[3, idx, 1] += ShNorm::N1 * x * colorGrad[1];
    shCoefs._grad_out[3, idx, 2] += ShNorm::N1 * x * colorGrad[2];

    directionGrad.x += dot(colorGrad, ShNorm::N1 * float3(shCoefs[3, idx, 0], shCoefs[3, idx, 1], shCoefs[3, idx, 2]));
    directionGrad.y += dot(colorGrad, -1. * ShNorm::N1 * float3(shCoefs[1, idx, 0], shCoefs[1, idx, 1], shCoefs[1, idx, 2]));
    directionGrad.z += dot(colorGrad, ShNorm::N1 * float3(shCoefs[2, idx, 0], shCoefs[2, idx, 1], shCoefs[2, idx, 2]));

    if (maxDeg < 2) {
        direction = diffPair(direction.p, directionGrad);
        return;
    }
    shCoefs._grad_out[4, idx, 0] += ShNorm::N2_m2 * xy * colorGrad[0];
    shCoefs._grad_out[4, idx, 1] += ShNorm::N2_m2 * xy * colorGrad[1];
    shCoefs._grad_out[4, idx, 2] += ShNorm::N2_m2 * xy * colorGrad[2];
    
    shCoefs._grad_out[5, idx, 0] += ShNorm::N2_m1 * yz * colorGrad[0];
    shCoefs._grad_out[5, idx, 1] += ShNorm::N2_m1 * yz * colorGrad[1];
    shCoefs._grad_out[5, idx, 2] += ShNorm::N2_m1 * yz * colorGrad[2];
    
    shCoefs._grad_out[6, idx, 0] += ShNorm::N2_0 * (2*z2-x2-y2) * colorGrad[0];
    shCoefs._grad_out[6, idx, 1] += ShNorm::N2_0 * (2*z2-x2-y2) * colorGrad[1];
    shCoefs._grad_out[6, idx, 2] += ShNorm::N2_0 * (2*z2-x2-y2) * colorGrad[2];
    
    shCoefs._grad_out[7, idx, 0] += ShNorm::N2_p1 * xz * colorGrad[0];
    shCoefs._grad_out[7, idx, 1] += ShNorm::N2_p1 * xz * colorGrad[1];
    shCoefs._grad_out[7, idx, 2] += ShNorm::N2_p1 * xz * colorGrad[2];
    
    shCoefs._grad_out[8, idx, 0] += ShNorm::N2_p2 * (x2-y2) * colorGrad[0];
    shCoefs._grad_out[8, idx, 1] += ShNorm::N2_p2 * (x2-y2) * colorGrad[1];
    shCoefs._grad_out[8, idx, 2] += ShNorm::N2_p2 * (x2-y2) * colorGrad[2];

    let shCoef4 = float3(shCoefs[4, idx, 0], shCoefs[4, idx, 1], shCoefs[4, idx, 2]);
    let shCoef5 = float3(shCoefs[5, idx, 0], shCoefs[5, idx, 1], shCoefs[5, idx, 2]);
    let shCoef6 = float3(shCoefs[6, idx, 0], shCoefs[6, idx, 1], shCoefs[6, idx, 2]);
    let shCoef7 = float3(shCoefs[7, idx, 0], shCoefs[7, idx, 1], shCoefs[7, idx, 2]);
    let shCoef8 = float3(shCoefs[8, idx, 0], shCoefs[8, idx, 1], shCoefs[8, idx, 2]);
    directionGrad.x += dot(colorGrad,
        y * ShNorm::N2_m2 * shCoef4
        - 2 * x * ShNorm::N2_0 * shCoef6
        + z * ShNorm::N2_p1 * shCoef7
        + 2 * x * ShNorm::N2_p2 * shCoef8);
    directionGrad.y += dot(colorGrad,
        x * ShNorm::N2_m2 * shCoef4
        + z * ShNorm::N2_m1 * shCoef5
        - 2 * y * ShNorm::N2_0 * shCoef6
        - 2 * y * ShNorm::N2_p2 * shCoef8);
    directionGrad.z += dot(colorGrad,
        y * ShNorm::N2_m1 * shCoef5
        + 4 * z * ShNorm::N2_0 * shCoef6
        + x * ShNorm::N2_p1 * shCoef7);

    if (maxDeg < 3) {
        direction = diffPair(direction.p, directionGrad);
        return;
    }
    shCoefs._grad_out[9, idx, 0] += ShNorm::N3_m3 * y*(3*x2-y2) * colorGrad[0];
    shCoefs._grad_out[9, idx, 1] += ShNorm::N3_m3 * y*(3*x2-y2) * colorGrad[1];
    shCoefs._grad_out[9, idx, 2] += ShNorm::N3_m3 * y*(3*x2-y2) * colorGrad[2];

    shCoefs._grad_out[10, idx, 0] += ShNorm::N3_m2 * xyz * colorGrad[0];
    shCoefs._grad_out[10, idx, 1] += ShNorm::N3_m2 * xyz * colorGrad[1];
    shCoefs._grad_out[10, idx, 2] += ShNorm::N3_m2 * xyz * colorGrad[2];

    shCoefs._grad_out[11, idx, 0] += ShNorm::N3_m1 * y*(4*z2-x2-y2) * colorGrad[0];
    shCoefs._grad_out[11, idx, 1] += ShNorm::N3_m1 * y*(4*z2-x2-y2) * colorGrad[1];
    shCoefs._grad_out[11, idx, 2] += ShNorm::N3_m1 * y*(4*z2-x2-y2) * colorGrad[2];

    shCoefs._grad_out[12, idx, 0] += ShNorm::N3_0 * z*(2*z2-3*x2-3*y2) * colorGrad[0];
    shCoefs._grad_out[12, idx, 1] += ShNorm::N3_0 * z*(2*z2-3*x2-3*y2) * colorGrad[1];
    shCoefs._grad_out[12, idx, 2] += ShNorm::N3_0 * z*(2*z2-3*x2-3*y2) * colorGrad[2];

    shCoefs._grad_out[13, idx, 0] += ShNorm::N3_p1 * x*(4*z2-x2-y2) * colorGrad[0];
    shCoefs._grad_out[13, idx, 1] += ShNorm::N3_p1 * x*(4*z2-x2-y2) * colorGrad[1];
    shCoefs._grad_out[13, idx, 2] += ShNorm::N3_p1 * x*(4*z2-x2-y2) * colorGrad[2];

    shCoefs._grad_out[14, idx, 0] += ShNorm::N3_p2 * z*(x2-y2) * colorGrad[0];
    shCoefs._grad_out[14, idx, 1] += ShNorm::N3_p2 * z*(x2-y2) * colorGrad[1];
    shCoefs._grad_out[14, idx, 2] += ShNorm::N3_p2 * z*(x2-y2) * colorGrad[2];

    shCoefs._grad_out[15, idx, 0] += ShNorm::N3_p3 * x*(x2-3*y2) * colorGrad[0];
    shCoefs._grad_out[15, idx, 1] += ShNorm::N3_p3 * x*(x2-3*y2) * colorGrad[1];
    shCoefs._grad_out[15, idx, 2] += ShNorm::N3_p3 * x*(x2-3*y2) * colorGrad[2];

    let shCoef9 = float3(shCoefs[9, idx, 0], shCoefs[9, idx, 1], shCoefs[9, idx, 2]);
    let shCoef10 = float3(shCoefs[10, idx, 0], shCoefs[10, idx, 1], shCoefs[10, idx, 2]);
    let shCoef11 = float3(shCoefs[11, idx, 0], shCoefs[11, idx, 1], shCoefs[11, idx, 2]);
    let shCoef12 = float3(shCoefs[12, idx, 0], shCoefs[12, idx, 1], shCoefs[12, idx, 2]);
    let shCoef13 = float3(shCoefs[13, idx, 0], shCoefs[13, idx, 1], shCoefs[13, idx, 2]);
    let shCoef14 = float3(shCoefs[14, idx, 0], shCoefs[14, idx, 1], shCoefs[14, idx, 2]);
    let shCoef15 = float3(shCoefs[15, idx, 0], shCoefs[15, idx, 1], shCoefs[15, idx, 2]);
    directionGrad.x += dot(colorGrad,
        6 * xy * ShNorm::N3_m3 * shCoef9
        + yz * ShNorm::N3_m2 * shCoef10
        - 2 * xy * ShNorm::N3_m1 * shCoef11
        - 6 * xz * ShNorm::N3_0 * shCoef12
        + (4 * z2 - 3 * x2 - y2) * ShNorm::N3_p1 * shCoef13
        + 2 * xz * ShNorm::N3_p2 * shCoef14
        + (3 * x2 - 3 * y2) * ShNorm::N3_p3 * shCoef15);
    directionGrad.y += dot(colorGrad,
        (3 * x2 - 3 * y2) * ShNorm::N3_m3 * shCoef9
        + xz * ShNorm::N3_m2 * shCoef10
        + (4 * z2 - x2 - 3 * y2) * ShNorm::N3_m1 * shCoef11
        - 6 * yz * ShNorm::N3_0 * shCoef12
        - 2 * xy * ShNorm::N3_p1 * shCoef13
        - 2 * yz * ShNorm::N3_p2 * shCoef14
        - 6 * xy * ShNorm::N3_p3 * shCoef15);
    directionGrad.z += dot(colorGrad,
        xy * ShNorm::N3_m2 * shCoef10
        + 8 * yz * ShNorm::N3_m1 * shCoef11
        + (6 * z2 - 3 * x2 - 3 * y2) * ShNorm::N3_0 * shCoef12
        + 8 * xz * ShNorm::N3_p1 * shCoef13
        + (x2 - y2) * ShNorm::N3_p2 * shCoef14);

    direction = diffPair(direction.p, directionGrad);
}

// sphere-frustrum test
bool isCulled(float3 cameraPos, float3 scales, float4 halfFovSinCos, float2 nearFar) {
    if (cameraPos.z < nearFar.x || cameraPos.z > nearFar.y || !isValidFloat(cameraPos)) return true;
    
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

// [ a b ]
// [ c d ]
float2 compute2x2Eigenvalues(float2x2 M) {
    let aMd = M._11 - M._22;
    let discrim = aMd * aMd + 4 * M._12 * M._21;
    let aPdHalf = (M._11 + M._22) / 2.;
    let discrimRootHalf = sqrt(discrim) / 2.;

    return float2(aPdHalf + discrimRootHalf, aPdHalf - discrimRootHalf);
}

[BackwardDerivative(projectGaussiansBwd)]
void projectGaussians(
    uint idx,
    DiffTensor<float, 2> worldPositions,
    DiffTensor<float, 2> rotations,
    DiffTensor<float, 2> scales,
    DiffTensor<float, 3> shCoefficients,
    int2 tilesXY,
    no_diff float4 intrinsics,
    no_diff float4 halfFovSinCos,
    no_diff float3x4 extrinsics,
    no_diff float2 nearFar,
    bool useExp,
    no_diff float colorBias,
    int maxShDegree,
    bool saveBackpropData,
    bool collectDensificationData,
    WDiffTensor<float, 2> screenPositions,
    IWTensor<float, 1> depths,
    WDiffTensor<float, 2> covariances,
    WDiffTensor<float, 2> colors,
    TensorView<int> counts,
    TensorView<bool> areColorsClamped,
    TensorView<float> gaussianImageRadii,
    TensorView<int> viewCounters)
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
    if (!isValidFloat(screenPos)) {
        counts[idx] = 0;
        return;
    }

    let rotation = float4(rotations[idx, 0], rotations[idx, 1], rotations[idx, 2], rotations[idx, 3]);
    let covarianceMat = compute2dCovariance(scale, rotation, (float3x3)extrinsics, getProjectionApproximation(cameraPos, intrinsics.xy));
    let regularizationTerm = .3;
    let covariance = float3(covarianceMat._11_22 + regularizationTerm, covarianceMat._12);
    if (!isValidFloat(covariance)) {
        counts[idx] = 0;
        return;
    }

    let intersectionNum = countIntersections(screenPos, covariance, tilesXY);
    if (intersectionNum == 0) {
        counts[idx] = 0;
        return;
    }

    let direction = getDirection(getCameraCenter(extrinsics), worldPos);
    let color = computeColor(direction, shCoefficients, idx, maxShDegree) + colorBias;
    let clampedColor = clamp(color, 0., 1.);
    
    screenPositions[idx, 0] = screenPos.x;
    screenPositions[idx, 1] = screenPos.y;
    depths[idx] = cameraPos.z;
    covariances[idx, 0] = covariance.x;
    covariances[idx, 1] = covariance.y;
    covariances[idx, 2] = covariance.z;
    colors[idx, 0] = clampedColor.x;
    colors[idx, 1] = clampedColor.y;
    colors[idx, 2] = clampedColor.z;
    counts[idx] = intersectionNum;

    if (!saveBackpropData) return;
    let isColorClamped = color != clampedColor;
    areColorsClamped[idx, 0] = isColorClamped.x;
    areColorsClamped[idx, 1] = isColorClamped.y;
    areColorsClamped[idx, 2] = isColorClamped.z;

    if (!collectDensificationData) return;
    let gaussianImageRadius = 3. * sqrt(compute2x2Eigenvalues(covarianceMat).x);
    gaussianImageRadii[idx] = max(gaussianImageRadii[idx], gaussianImageRadius);
    viewCounters[idx] = viewCounters[idx] + 1;
}

void projectGaussiansBwd(
    uint idx,
    DiffTensor<float, 2> worldPositions,
    DiffTensor<float, 2> rotations,
    DiffTensor<float, 2> scales,
    DiffTensor<float, 3> shCoefficients,
    int2 tilesXY,
    float4 intrinsics,
    float4 halfFovSinCos,
    float3x4 extrinsics,
    float2 nearFar,
    bool useExp,
    float colorBias,
    int maxShDegree,
    bool saveBackpropData,
    bool collectDensificationData,
    WDiffTensor<float, 2> screenPositions,
    IWTensor<float, 1> depths,
    WDiffTensor<float, 2> covariances,
    WDiffTensor<float, 2> colors,
    TensorView<int> counts,
    TensorView<bool> areColorsClamped,
    TensorView<float> gaussianImageRadii,
    TensorView<int> viewCounters)
{
    if (!saveBackpropData || idx >= worldPositions.shape[0]) return;
    if (counts[idx] <= 0) return;

    let screenPosGrad = float2(screenPositions._grad_in[idx, 0], screenPositions._grad_in[idx, 1]);
    let covariancesGrad = float3(covariances._grad_in[idx, 0], covariances._grad_in[idx, 1], covariances._grad_in[idx, 2]);
    let covMatGrad = float2x2(covariancesGrad.x, covariancesGrad.z / 2., covariancesGrad.z / 2., covariancesGrad.y);
    let colorGrad = float3(
        colors._grad_in[idx, 0] * (areColorsClamped[idx, 0] ? .0 : 1.),
        colors._grad_in[idx, 1] * (areColorsClamped[idx, 1] ? .0 : 1.),
        colors._grad_in[idx, 2] * (areColorsClamped[idx, 2] ? .0 : 1.)
    );

    var worldPos = diffPair(float3(worldPositions[idx, 0], worldPositions[idx, 1], worldPositions[idx, 2]));
    var cameraPos = diffPair(mul(extrinsics, float4(worldPos.p, 1.)));
    bwd_diff(cameraToScreen)(cameraPos, intrinsics, screenPosGrad);
    var camPosGradAccum = cameraPos.d;

    var storedScale = float3(scales[idx, 0], scales[idx, 1], scales[idx, 2]);
    var scale = diffPair(useExp ? exp(storedScale) : storedScale);
    var rotation = diffPair(float4(rotations[idx, 0], rotations[idx, 1], rotations[idx, 2], rotations[idx, 3]));
    var projApprox = diffPair(getProjectionApproximation(cameraPos.p, intrinsics.xy));
    let cameraRotation = (float3x3)extrinsics;
    bwd_diff(compute2dCovariance)(scale, rotation, cameraRotation, projApprox, covMatGrad);
    bwd_diff(getProjectionApproximation)(cameraPos, intrinsics.xy, projApprox.d);
    let scaleGrad = useExp ? scale.d * scale.p : scale.d;
    camPosGradAccum += cameraPos.d;
    let worldPosGradCamPosContrib = mul(camPosGradAccum, cameraRotation);

    let cameraCenter = getCameraCenter(extrinsics);
    var viewDir = diffPair(getDirection(cameraCenter, worldPos.p));
    bwd_diff(computeColor)(viewDir, shCoefficients, idx, maxShDegree, colorGrad);
    bwd_diff(getDirection)(cameraCenter, worldPos, viewDir.d);
    let worldPosGrad = worldPosGradCamPosContrib + worldPos.d;

    worldPositions._grad_out[idx, 0] += worldPosGrad.x;
    worldPositions._grad_out[idx, 1] += worldPosGrad.y;
    worldPositions._grad_out[idx, 2] += worldPosGrad.z;
    rotations._grad_out[idx, 0] += rotation.d.x;
    rotations._grad_out[idx, 1] += rotation.d.y;
    rotations._grad_out[idx, 2] += rotation.d.z;
    rotations._grad_out[idx, 3] += rotation.d.w;
    scales._grad_out[idx, 0] += scaleGrad.x;
    scales._grad_out[idx, 1] += scaleGrad.y;
    scales._grad_out[idx, 2] += scaleGrad.z;
}

void createGaussianInstances(
    uint idx,
    uint gaussianCount,
    ITensor<int, 1> counts,
    ITensor<int, 1> cumulativeCounts,
    ITensor<float, 2> positions,
    ITensor<float, 1> depths,
    ITensor<float, 2> covariances,
    int2 tilesXY,
    IWTensor<int, 1> instances,
    IWTensor<int64_t, 1> keys)
{
    if (idx >= gaussianCount) return;
    let count = counts[idx];
    if (count == 0) return;

    let offset = cumulativeCounts[idx] - count;
    for (int i = 0; i < count; i++) {
        instances[offset + i] = idx;
    }

    let position = float2(positions[idx, 0], positions[idx, 1]);
    let variances = float2(covariances[idx, 0], covariances[idx, 1]);
    let depth = depths[idx];

    let halfSize = 3. * sqrt(variances);
    let minPos = position - halfSize;
    let maxPos = position + halfSize;
    let minTileCoords = max(int2(0, 0), int2(floor(minPos / TILE_SIZE)));
    let maxTileCoords = min(tilesXY - 1, int2(floor(maxPos / TILE_SIZE))) + 1;

    int currInstance = 0;
    for (int j = minTileCoords.y; j < maxTileCoords.y; j++) {
        for (int i = minTileCoords.x; i < maxTileCoords.x; i++) {
            keys[offset + currInstance] = reinterpret<int64_t>(createKey(getTileId(i, j, tilesXY), depth));
            currInstance++;
        }
    }
}

void findTileRangesNoShuffle(uint idx, uint keyCount, ITensor<int64_t, 1> keys, IWTensor<int, 2> ranges) {
    let lastIdx = keyCount - 2;
    if (idx > lastIdx) return;
    
    let nextIdx = idx + 1;
    let prevTileId = extractTileId(keys[idx]);
    let nextTileId = extractTileId(keys[nextIdx]);
    if (idx == 0) ranges[prevTileId, 0] = 0;
    if (idx == lastIdx) ranges[nextTileId, 1] = keyCount;
    if (prevTileId == nextTileId) return;
    ranges[prevTileId, 1] = nextIdx;
    ranges[nextTileId, 0] = nextIdx;
}

void findTileRanges(uint idx, uint keyCount, ITensor<int64_t, 1> keys, IWTensor<int, 2> ranges) {
    if (idx >= keyCount) return;

    let tileId = extractTileId(keys[idx]);
    var prevTileId = WaveReadLaneAt(tileId, WaveGetLaneIndex() - 1);
    if (WaveIsFirstLane()) {
        prevTileId = (idx == 0) ? tileId : extractTileId(keys[idx - 1]);
    }

    if (idx == 0) ranges[tileId, 0] = 0;
    if (idx == keyCount - 1) ranges[tileId, 1] = keyCount;
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
    int2 screensize,
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
    no_diff float3 backgroundColor,
    WDiffTensor<float, 3> image,
    TensorView<float> accumulatedTransmittances,
    TensorView<int> processedGaussianCount)
{
    let workgroupSize = WorkgroupSize().x;
    let tileId = threadId / workgroupSize; // here we assume that each tile is mapped to a workgroup
    let localThreadId = threadId % workgroupSize;
    let pixelCoords = getPixelCoords(localThreadId, tileId, tilesXY.x);
    let pixelCenter = pixelCoords + .5;
    let isValidPixel = all(pixelCoords < screensize);
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
                        break;
                    }
                    let alpha = min(computeGaussianAlpha(i, pixelCenter, useSig), maxAlpha);
                    if (alpha <= alphaThres) continue;

                    let nextTransmittance = transmittance * (1 - alpha);
                    if (nextTransmittance <= minTransmittance) {
                        hasFinished = true;
                        break;
                    }
                    let gaussianColor = float3(sharedGaussians.reds[i], sharedGaussians.greens[i], sharedGaussians.blues[i]);
                    color += gaussianColor * alpha * transmittance;
                    transmittance = nextTransmittance;
                }
                if (hasFinished || processedGaussianNum == maxGaussians) {
                    hasFinished = true;
                    finishedThreadNum.increment();
                }
            }
            loadIdx += workgroupSize;

            GroupMemoryBarrierWithGroupSync();
        } while (finishedThreadNum.load() != workgroupSize);
    }

    if (!isValidPixel) return;
    color += transmittance * backgroundColor;
    image[pixelCoords.y, pixelCoords.x, 0] = color.x;
    image[pixelCoords.y, pixelCoords.x, 1] = color.y;
    image[pixelCoords.y, pixelCoords.x, 2] = color.z;

    if (!saveBackpropData) return;
    accumulatedTransmittances[pixelCoords.y, pixelCoords.x] = transmittance;
    processedGaussianCount[pixelCoords.y, pixelCoords.x] = processedGaussianNum;
}

groupshared int sharedBlockMaxGaussians;

int getBlockMaxGaussians(uint localThreadId, int threadMaxGaussians) {
    if (localThreadId == 0) sharedBlockMaxGaussians = 0;
    GroupMemoryBarrierWithGroupSync();

    let waveMaxGaussians = WaveActiveMax(threadMaxGaussians);
    if (WaveIsFirstLane()) InterlockedMax(sharedBlockMaxGaussians, waveMaxGaussians);
    GroupMemoryBarrierWithGroupSync();

    return sharedBlockMaxGaussians;
}

void renderGaussiansBwd(
    uint2, uint threadId,
    int2 screensize,
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
    float minTransmittance,
    bool saveBackpropData,
    float3 backgroundColor,
    WDiffTensor<float, 3> image,
    TensorView<float> accumulatedTransmittances,
    TensorView<int> processedGaussianCount)
{
    if (!saveBackpropData) return;

    let workgroupSize = WorkgroupSize().x;
    let tileId = threadId / workgroupSize;
    let localThreadId = threadId % workgroupSize;
    let pixelCoords = getPixelCoords(localThreadId, tileId, tilesXY.x);
    let pixelCenter = pixelCoords + .5;
    let isValidPixel = all(pixelCoords < screensize);
    let tileRange = int2(tileRanges[tileId, 0], tileRanges[tileId, 1]);
    if (tileRange.y <= tileRange.x) return; // no gaussians were assigned to the tile/block, can return

    let threadMaxGaussians = isValidPixel ? processedGaussianCount[pixelCoords.y, pixelCoords.x] : 0;
    let blockMaxGaussians = getBlockMaxGaussians(localThreadId, threadMaxGaussians);
    if (blockMaxGaussians <= 0) return; // no thread in the tile/block has processed any gaussians, can return

    int loadIdx = tileRange.x + blockMaxGaussians - 1 - localThreadId;
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
    let finalTransm = isValidPixel ? accumulatedTransmittances[pixelCoords.y, pixelCoords.x] : 1.;
    float accumTransm = finalTransm;
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
                accumTransm = accumTransm / (1. - alpha);
                let dPixeldColor = float3(alpha * accumTransm);
                let color = float3(sharedGaussians.reds[i], sharedGaussians.greens[i], sharedGaussians.blues[i]);
                let dPixeldAlpha = color * accumTransm - backSum / (1. - alpha) - finalTransm / (1. - alpha) * backgroundColor;
                let dAlphadOpacity = eSig;
                let dAlphadSig = -alpha;
                let dSigdMean = invSigmaDelta;
                let dSigdSigma = -.5 * float3(
                    invSigmaDelta.x * invSigmaDelta.x,
                    invSigmaDelta.y * invSigmaDelta.y,
                    2. * invSigmaDelta.x * invSigmaDelta.y
                );

                let colorGrad = dPixeldColor * pixelGrad;
                let alphaGrad = dot(dPixeldAlpha, pixelGrad);
                let opacityGrad = (useSig ? opacity * (1. - opacity) : 1.) * dAlphadOpacity * alphaGrad;
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
        GroupMemoryBarrierWithGroupSync();
    }
}
