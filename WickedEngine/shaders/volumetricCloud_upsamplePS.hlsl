#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> cloud_current : register(t0);
Texture2D<float2> cloud_depth_current : register(t1);

static const int UPSAMPLE_SAMPLE_RADIUS = 1;

#define GAUSSIAN_SIGMA_SPATIAL 0.5
// GGMAX 1.66: the range kernel is a DISTANCE tolerance. Stock 100.0 assumed meter world
// units; GG world units are inches, so 100 units = 2.5 real meters — the bilateral blend
// rejected valid same-surface taps at every terrain/sky discontinuity and painted the
// bright "mountain silhouette line". 3937 units = the stock 100 m intent.
#define GAUSSIAN_SIGMA_RANGE 3937.0

#define UPSAMPLE_TOLERANCE 0.15

// GGMAX 1.66: full float — the half version overflowed on both fronts in an inch-unit
// world: 2*sigma*sigma > 65504 (half inf -> x*x/inf = NaN for far taps) and x itself
// (depth differences reach millions of units).
float Gaussian(float x, float sigma)
{
	return exp(-x * x / (2.0 * sigma * sigma));
}

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
	const uint2 pixel = pos.xy;
	const float depth = texture_depth[pixel];
	
	const uint2 reprojectionCoord = pixel / 2;
	
	const float2 reprojectionResolution = postprocess.params0.xy;
	const float2 reprojectionTexelSize = postprocess.params0.zw;

	float3 depthWorldPosition = reconstruct_position(uv, depth);
	float tToDepthBuffer = length(depthWorldPosition - GetCamera().position);
	
	// If sky, set distance to infinite
	tToDepthBuffer = depth == 0.0 ? FLT_MAX : tToDepthBuffer;

	// Adapted from upsample_bilateral_float4CS:
	
	const float2 uv00 = uv - reprojectionTexelSize * 0.5;
	const float2 uv10 = float2(uv00.x + reprojectionTexelSize.x, uv00.y);
	const float2 uv01 = float2(uv00.x, uv00.y + reprojectionTexelSize.y);
	const float2 uv11 = float2(uv00.x + reprojectionTexelSize.x, uv00.y + reprojectionTexelSize.y);

	const float4 lineardepth_lowres = float4(
		cloud_depth_current.SampleLevel(sampler_point_clamp, uv00, 0).g,
		cloud_depth_current.SampleLevel(sampler_point_clamp, uv10, 0).g,
		cloud_depth_current.SampleLevel(sampler_point_clamp, uv01, 0).g,
		cloud_depth_current.SampleLevel(sampler_point_clamp, uv11, 0).g
	);

	const float4 depthDiff = abs(tToDepthBuffer - lineardepth_lowres);	
	float depthDiffMax = max(max(depthDiff.x, depthDiff.y), max(depthDiff.z, depthDiff.w));
	
	half4 result = 0;
	
	[branch]
	if (depthDiffMax < tToDepthBuffer * 0.2)
	{
		// small error, take bilinear sample:
		result = cloud_current.SampleLevel(sampler_linear_clamp, uv, 0);
	}
	else
	{
		// large error, calculate weight and color depending on depth difference with gaussian configuration
		half4 color = 0;
		float weightSum = 0; // Note: weights need full precision on Nvidia Vulkan!
		float nearestTapSurface = FLT_MAX;   // GGMAX 3.76, see the fallback below
				
		[unroll]
		for (int y = -UPSAMPLE_SAMPLE_RADIUS; y <= UPSAMPLE_SAMPLE_RADIUS; y++)
		{
			[unroll]
			for (int x = -UPSAMPLE_SAMPLE_RADIUS; x <= UPSAMPLE_SAMPLE_RADIUS; x++)
			{
				int2 offset = int2(x, y);
			
				int2 neighborReprojectionCoord = reprojectionCoord + offset;
				float2 neighborReprojectionUV = (neighborReprojectionCoord + 0.5) / reprojectionResolution;
			
				half4 cloudResult = cloud_current.SampleLevel(sampler_linear_clamp, neighborReprojectionUV, 0);
				// GGMAX 1.66: read the cloud depth at FULL precision — the half read overflowed
				// to +inf beyond 65504 units (1.66 real km here), zeroing every tap weight at
				// distance and leaving un-clouded holes along depth edges.
				float cloudDepth = cloud_depth_current[neighborReprojectionCoord].g;
				// GGMAX 3.76: .g is the distance to the GEOMETRY that tap's ray hit (FLT_MAX for
				// sky). Keep the closest one - the fallback below needs to know whether every
				// rejected tap was looking PAST this pixel's own surface.
				nearestTapSurface = min( nearestTapSurface, cloudDepth );
				
				float spatialWeight = Gaussian(length(float2(offset)), GAUSSIAN_SIGMA_SPATIAL);
				float rangeWeight = Gaussian(abs(tToDepthBuffer - cloudDepth), GAUSSIAN_SIGMA_RANGE);
				float weight = spatialWeight * rangeWeight;
				
				color += cloudResult * weight;
				weightSum += weight;
			}
		}

		if (weightSum > 0)
		{
			result = color / weightSum;
		}
		else
		{
			// GGMAX 1.66: never emit a hard zero (an un-clouded hole) when every tap was
			// rejected — fall back to the bilinear sample like the small-error path.
			//
			// ★★★ GGMAX 3.76: ...UNLESS EVERY TAP WAS LOOKING PAST US.
			//
			// "Rejected" has two causes and they want opposite answers. A tap that saw a surface
			// NEARER than this pixel found cloud genuinely in front of us, and 1.66's fallback is
			// right. A tap that saw one FARTHER found cloud BEHIND our own surface, and painting
			// that is how a distant tree billboard ends up wearing a sky cloud.
			//
			// It FLICKERS, which is why it matters more than it reads. The raymarch is a
			// quarter-res checkerboard (volumetricCloud_renderCS.hlsl: subPixelIndex =
			// volumetricclouds_frame % 4), so one occluder sample covers 4x4 full-res pixels and
			// WHICH pixel it lands on rotates every frame. Along a thin silhouette the taps flip
			// between hitting the tree and seeing sky past it, so the fallback fires on some
			// frames and not others - which reads as shimmer, the worst kind of wrong.
			//
			// The 1% margin keeps float equality on a same-surface tap out of the behind case.
			if ( nearestTapSurface > tToDepthBuffer * 1.01 )
			{
				result = 0;   // all taps' cloud is behind our surface - contribute nothing
			}
			else
			{
				result = cloud_current.SampleLevel(sampler_linear_clamp, uv, 0);
			}
		}
	}

	return result;
}
