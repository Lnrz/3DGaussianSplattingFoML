module math;

public static const float TAU = 6.2831853;

namespace ShNorm {
    public static const float N0 = 0.28209479177387814;

    public static const float N1 = 0.4886025119029199;

    public static const float N2_m2 = 1.0925484305920792;
    public static const float N2_m1 = -1.0925484305920792;
    public static const float N2_0 = 0.31539156525252005;
    public static const float N2_p1 = -1.0925484305920792;
    public static const float N2_p2 = 0.5462742152960396;

    public static const float N3_m3 = -0.5900435899266435;
    public static const float N3_m2 = 2.890611442640554;
    public static const float N3_m1 = -0.4570457994644658;
    public static const float N3_0 = 0.3731763325901154;
    public static const float N3_p1 = -0.4570457994644658;
    public static const float N3_p2 = 1.445305721320277;
    public static const float N3_p3 = -0.5900435899266435;
}

// Convert quaternion to rotation matrix
//
// Quaternion must be normalized and in WXYZ order
[Differentiable]
public float3x3 quaternionToMatrix(float4 quaternion) {
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

// Return true if no value in x is NaN or inf
public bool isValidFloat<T, let n : int>(vector<T, n> x) where T : __BuiltinFloatingPointType {
    return all(!isnan(x) && !isinf(x));
}

// Compute the eigenvalues of a 2x2 matrix
//
// Return the bigger eigenvalue first, the smaller second
public float2 compute2x2Eigenvalues(float2x2 M) {
    // Assuming the matrix to be
    // [ a b ]
    // [ c d ]
    let aMd = M._11 - M._22;
    let discrim = aMd * aMd + 4 * M._12 * M._21;
    let aPdHalf = (M._11 + M._22) / 2.;
    let discrimRootHalf = sqrt(discrim) / 2.;

    return float2(aPdHalf + discrimRootHalf, aPdHalf - discrimRootHalf);
}

public T sigmoid<T>(T x) where T : __BuiltinFloatingPointType {
    return T(1.) / (T(1.) + exp(-x));
}
