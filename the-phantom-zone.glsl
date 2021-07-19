// The Phantom Zone
// by @P_Malin
// https://www.shadertoy.com/view/ftlXWB

// A silly shader designed to work with Memix https://www.memix.app/
// Save the shader as %LOCALAPPDATA%\Beautypi\Memix\shader.txt

// https://github.com/pmalin/pmalin-memix-shaders

// Thanks for the following:
// Dave Hoskins - hash function from https://www.shadertoy.com/view/4djSRW
// iq - soft shadow from https://www.shadertoy.com/view/lsKcDD 
// iq - box intersection from https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
// iq - smooth min from https://www.iquilezles.org/www/articles/smin/smin.htm

// Also see a similar shader from dean_the_coder: https://www.shadertoy.com/view/Wd2fzV

#define RAYMARCH_ITER 32
#define MAX_RAYMARCH_DIST 70.0
#define SHADOW_STEPS 16


vec3 boxDimensions = vec3(1,1,0.02);
vec3 glassColour = vec3( 0.6, 0.84, 0.9 );
float glassDensity = 40.0;

vec3 sunDir = normalize(vec3(0.3, 0.5, -0.2));
vec3 sunColor = vec3(1, 0.95, 0.9) * 1.7;
vec3 ambientColor = vec3(0.3, 0.7, 1.0) * 1.0;

float fogDensity= 0.00005;
float fogHeightFalloff = 1.0;

float sliceBegin = -3.5f;
float sliceHeight = 1.0f;

#define PI 3.1415925654

vec3 RotateX( vec3 pos, float angle )
{
    float s = sin(angle);
    float c = cos(angle);
    
    return vec3( pos.x, c * pos.y + s * pos.z, -s * pos.y + c * pos.z);
}

vec3 RotateY( vec3 pos, float angle )
{
    float s = sin(angle);
    float c = cos(angle);
    
    return vec3( c * pos.x + s * pos.z, pos.y, -s * pos.x + c * pos.z);
}

vec3 RotateZ( vec3 pos, float angle )
{
    float s = sin(angle);
    float c = cos(angle);
    
    return vec3( c * pos.x + s * pos.y, -s * pos.x + c * pos.y, pos.z);
}

struct Ray
{
    vec3 pos;
    vec3 dir;
};

struct CameraState
{
    vec3 pos;
    vec3 target;
    vec3 up;
};

mat3 GetCameraMatrix( CameraState cameraState )
{
    vec3 zDir = normalize(cameraState.target - cameraState.pos);    
    vec3 xDir = normalize( cross( cameraState.up, zDir ) );
    vec3 yDir = normalize( cross( zDir, xDir ) );
    
    mat3 mat = mat3( xDir, yDir, zDir );
    
    return mat;
}

Ray GetCameraRay( vec2 coord, CameraState cameraState )
{
    vec3 viewDir = normalize(vec3( coord.xy, 1.0f ));
    
    mat3 mat = GetCameraMatrix( cameraState );
    
    Ray ray = Ray( cameraState.pos, mat * viewDir );
    
    return ray;
}

#define MAT_NONE -1
#define MAT_DEFAULT 0
#define MAT_QUAD 1


// hash from https://www.shadertoy.com/view/4djSRW
// Hash without Sine
// by Dave_Hoskins

//  1 out, 1 in...
float hash11(float p)
{
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

//  3 out, 2 in...
vec3 hash32(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy+p3.yzz)*p3.zyx);
}


float SmoothNoise( vec2 o ) 
{
    //vec2 p = floor(o);
    vec2 f = fract(o);
    vec2 p = o-f;
        
    float n = p.x + p.y*57.0;

    float a = hash11(n+  0.0);
    float b = hash11(n+  1.0);
    float c = hash11(n+ 57.0);
    float d = hash11(n+ 58.0);
    
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    
    vec2 t = 3.0 * f2 - 2.0 * f3;
    
    float u = t.x;
    float v = t.y;

    float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
    return res;
}

float CircleBomb( vec2 pos, float range )
{   
    float dist = MAX_RAYMARCH_DIST;
    for(float yo=-range; yo<=range; yo+=1.0)
    {
        for(float xo=-range; xo<=range; xo+=1.0)
        {
            vec2 cellPos = pos + vec2(xo,yo);
            vec2 cellIndex = floor(cellPos);
            
            vec3 hash = hash32( cellIndex );
            
            vec2 circleOffset = hash.xy * 0.5;
            
            vec2 circlePos = cellIndex + 0.5 + circleOffset;
            
            float circleRadius = hash.z * 0.5;
            float circleDist = length( circlePos - pos ) - circleRadius;
            
            if ( circleDist < dist )
            {
                dist = circleDist;
            }
        }
    }
    
    return dist;
}

float GetSlice( float h )
{
    return floor( (h-sliceBegin) / sliceHeight );
}

struct RaymarchResult
{
    float dist;
    int objectId;
    vec3 uvw;
};

RaymarchResult Scene_GetDistance( vec3 pos );

RaymarchResult Scene_Raymarch( Ray ray, float minDist, float maxDist )
{   
    RaymarchResult result;
    result.dist = 0.0;
    result.uvw = vec3(0.0);
    result.objectId = MAT_NONE;
    
    float t = minDist;
    
    for(int i=0; i<RAYMARCH_ITER; i++)
    {       
        float epsilon = 0.000001 * t;
        result = Scene_GetDistance( ray.pos + ray.dir * t );
        if ( abs(result.dist) < epsilon )
        {
            break;
        }
                        
        if ( t > maxDist )
        {
            result.objectId = MAT_NONE;
            t = maxDist;
            break;
        }       
        
        if ( result.dist > 1.0 )
        {
            result.objectId = MAT_NONE;
        }    
        
        t += result.dist; 
    }
    
    result.dist = max( t, minDist );


    return result;
}


RaymarchResult Scene_Union( RaymarchResult a, RaymarchResult b )
{
    if ( b.dist < a.dist )
    {
        return b;
    }
    return a;
}

vec3 Scene_GetNormal(vec3 pos)
{
    const float fDelta = 0.0001;
    vec2 e = vec2( -1, 1 );
    
    vec3 vNormal =
        Scene_GetDistance( e.yxx * fDelta + pos ).dist * e.yxx + 
        Scene_GetDistance( e.xxy * fDelta + pos ).dist * e.xxy + 
        Scene_GetDistance( e.xyx * fDelta + pos ).dist * e.xyx + 
        Scene_GetDistance( e.yyy * fDelta + pos ).dist * e.yyy;
    
    return normalize( vNormal );
} 

struct TraceResult
{
    float dist;
    int objectId;
    vec3 uvw;
    vec3 pos;    
    vec3 normal;
};

vec3 BoxDomainRotate( vec3 pos );
vec3 BoxDomainInvRotate( vec3 pos );


// https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
// axis aligned box centered at the origin, with size boxSize
vec2 boxIntersection( in vec3 ro, in vec3 rd, vec3 boxSize, out vec3 outNormal ) 
{
    vec3 m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    vec3 n = m*ro;   // can precompute if traversing a set of aligned boxes
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    outNormal = -sign(rd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);
    return vec2( tN, tF );
}


TraceResult Scene_Raytrace( Ray ray, float minDist, float maxDist )
{
    TraceResult result;
    
    result.dist = MAX_RAYMARCH_DIST;
    result.objectId = MAT_NONE;

    vec3 boxRayDir = BoxDomainRotate( ray.dir );
    vec3 boxRayPos = BoxDomainRotate( ray.pos );    

    vec2 t = boxIntersection( boxRayPos, boxRayDir, boxDimensions, result.normal);

    result.normal = BoxDomainInvRotate( result.normal );

    result.uvw = (boxRayPos + boxRayDir * t.x);

    if ( t.x  >= 0.0 )
    {
        result.dist = t.x;
        result.objectId = MAT_QUAD;
    }
    
    return result;
}


TraceResult Scene_Trace( Ray ray, float minDist, float maxDist )
{
    TraceResult result;    

    TraceResult raytraceResult = Scene_Raytrace( ray, minDist, maxDist );
        
    if (ray.dir.y >= 0.0 )
    {
        result.dist = MAX_RAYMARCH_DIST;
        result.objectId = MAT_NONE;
        result.normal = -ray.dir;
        result.uvw = vec3(0);
        result.pos = ray.pos + ray.dir * result.dist;
    }
    else
    {
        float yStart = sliceBegin;
        
        float t = (yStart - ray.pos.y) / ray.dir.y;
        
        if ( t > minDist )
        {
            minDist = t;
        }
        
        RaymarchResult raymarchResult;
        
        raymarchResult = Scene_Raymarch( ray, minDist, maxDist );        

        result.dist = raymarchResult.dist;
        result.objectId = raymarchResult.objectId;
        result.uvw = raymarchResult.uvw;
        result.pos = ray.pos + ray.dir * result.dist;
        result.normal = Scene_GetNormal( result.pos );     
        
    }
        
    if ( raytraceResult.dist < result.dist )
    {
        result.dist = raytraceResult.dist;
        result.objectId = raytraceResult.objectId;
        result.uvw = raytraceResult.uvw;
        result.pos = ray.pos + ray.dir * result.dist;
        result.normal = raytraceResult.normal;             
    }

    return result;
}

float Scene_TraceShadow( Ray ray, float minDist, float lightDist )
{
    // Soft Shadow Variation
    // https://www.shadertoy.com/view/lsKcDD    
    // based on Sebastian Aaltonen's soft shadow improvement
    
    float res = 1.0;
    float t = minDist;
    float ph = 1e10; // big, such that y = 0 on the first iteration
        
    for( int i=0; i<SHADOW_STEPS; i++ )
    {
        float h = Scene_GetDistance( ray.pos + ray.dir * t ).dist;

        // use this if you are getting artifact on the first iteration, or unroll the
        // first iteration out of the loop
        //float y = (i==0) ? 0.0 : h*h/(2.0*ph); 

        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, 10.0*d/max(0.0,t-y) );
        ph = h;
        
        t += h;
        
        if( res < 0.0001 || t > lightDist ) break;
        
    }
    return clamp( res, 0.0, 1.0 );    
}

float Scene_GetAmbientOcclusion( Ray ray )
{
    float occlusion = 0.0;
    float scale = 1.0;
    for( int i=0; i<5; i++ )
    {
        float fOffsetDist = 0.001 + 0.05*float(i)/4.0;
        vec3 AOPos = ray.dir * fOffsetDist + ray.pos;
        float dist = Scene_GetDistance( AOPos ).dist;
        occlusion += (fOffsetDist - dist) * scale;
        scale *= 0.46;
    }
    
    return clamp( 1.0 - 30.0*occlusion, 0.0, 1.0 );
}

struct SurfaceInfo
{
    vec3 pos;
    vec3 normal;
    vec3 albedo;
    vec3 r0;
    float gloss;
    vec3 emissive;
};
    
SurfaceInfo Scene_GetSurfaceInfo( Ray ray, TraceResult traceResult );

struct SurfaceLighting
{
    vec3 diffuse;
    vec3 specular;
};

float Light_GIV( float dotNV, float k)
{
    return 1.0 / ((dotNV + 0.0001) * (1.0 - k)+k);
}

float AlphaSqrFromGloss( float gloss )
{
    float MAX_SPEC = 10.0;
    return 2.0f  / ( 2.0f + exp2( gloss * MAX_SPEC) );
}
    
void Light_Add( inout SurfaceLighting lighting, SurfaceInfo surface, vec3 viewDir, vec3 lightDir, vec3 lightColour )
{
    float NDotL = clamp(dot(lightDir, surface.normal), 0.0, 1.0);
    
    lighting.diffuse += lightColour * NDotL;

    if ( surface.gloss > 0.0 )
    {
        vec3 H = normalize( -viewDir + lightDir );
        float NdotV = clamp(dot(-viewDir, surface.normal), 0.0, 1.0);
        float NdotH = clamp(dot(surface.normal, H), 0.0, 1.0);

        // D

        float alphaSqr = AlphaSqrFromGloss( surface.gloss );
        float alpha = sqrt( alphaSqr );
        float denom = NdotH * NdotH * (alphaSqr - 1.0) + 1.0;
        float d = alphaSqr / (PI * denom * denom);

        float k = alpha / 2.0;
        float vis = Light_GIV( NDotL, k ) * Light_GIV( NdotV, k );

        float specularIntensity = d * vis * NDotL;    
        lighting.specular += lightColour * specularIntensity;    
    }
}

void Light_AddDirectional(inout SurfaceLighting lighting, SurfaceInfo surface, vec3 viewDir, vec3 lightDir, vec3 lightColour)
{   
    float attenuation = 1.0;
    Ray shadowRay;
    shadowRay.pos = surface.pos + surface.normal * 0.001;
    shadowRay.dir = lightDir;
    float shadowFactor = Scene_TraceShadow( shadowRay, 0.01, 10.0 );
    
    Light_Add( lighting, surface, viewDir, lightDir, lightColour * shadowFactor * attenuation);
}

SurfaceLighting Scene_GetSurfaceLighting( Ray ray, SurfaceInfo surfaceInfo );

vec3 Env_GetSkyColor( Ray ray );

vec3 Light_GetFresnel( vec3 view, vec3 normal, vec3 r0, float gloss )
{
    float NdotV = max( 0.0, dot( view, normal ) );

    return r0 + (vec3(1.0) - r0) * pow( 1.0 - NdotV, 5.0 ) * pow( gloss, 20.0 );
}

vec3 Env_ApplyAtmosphere( vec3 colour, Ray ray, float dist );


vec3 Scene_GetColour( Ray ray )
{
    vec3 resultColor = vec3(0.0);
            
    TraceResult firstTraceResult;
    
    float startDist = 0.0f;
    float maxDist = MAX_RAYMARCH_DIST;
    
    vec3 remaining = vec3(1.0);
    
    for( int passIndex=0; passIndex < 2; passIndex++ )
    {
        TraceResult traceResult = Scene_Trace( ray, startDist, maxDist );

        if ( passIndex == 0 )
        {
            firstTraceResult = traceResult;
        }
        
        vec3 colour = vec3(0);
        vec3 reflectAmount = vec3(0);
        
        if( traceResult.objectId < 0 )
        {
            colour = Env_GetSkyColor( ray );
            colour = Env_ApplyAtmosphere( colour, ray, traceResult.dist );
        }
        else
        {
            
            SurfaceInfo surfaceInfo = Scene_GetSurfaceInfo( ray, traceResult );
            SurfaceLighting surfaceLighting = Scene_GetSurfaceLighting( ray, surfaceInfo );
                
            if ( surfaceInfo.gloss <= 0.0 )
            {
                reflectAmount = vec3(0.0);
            }
            else
            {
                // calculate reflectance (Fresnel)
                reflectAmount = Light_GetFresnel( -ray.dir, surfaceInfo.normal, surfaceInfo.r0, surfaceInfo.gloss );            
            }
            
            colour = (surfaceInfo.albedo * surfaceLighting.diffuse + surfaceInfo.emissive) * (vec3(1.0) - reflectAmount); 
            
            vec3 reflectRayOrigin = surfaceInfo.pos;
            vec3 reflectRayDir = normalize( reflect( ray.dir, surfaceInfo.normal ) );
            startDist = 0.001 / max(0.0000001,abs(dot( reflectRayDir, surfaceInfo.normal ))); 

            colour += surfaceLighting.specular * reflectAmount;            

            colour = Env_ApplyAtmosphere( colour, ray, traceResult.dist );
            
            ray.pos = reflectRayOrigin;
            ray.dir = reflectRayDir;
        }
        
        resultColor += colour * remaining;
        remaining *= reflectAmount;                
        
        if ( (remaining.x + remaining.y + remaining.z) < 0.01 )
        {
            break;
        }            
    }
 
    return vec3( resultColor );
}

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

vec3 BoxDomainRotate( vec3 pos )
{
    pos = RotateX( pos, iTime );
    pos = RotateZ( pos, 0.5 );
    
    return pos;
}

vec3 BoxDomainInvRotate( vec3 pos )
{
    pos = RotateZ( pos, -0.5 );
    pos = RotateX( pos, -iTime );
    
    return pos;
}

// https://www.iquilezles.org/www/articles/smin/smin.htm
// polynomial smooth min (k = 0.1);
float sminCubic( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*h*k*(1.0/6.0);
}

RaymarchResult Scene_GetDistance( vec3 pos )
{
    RaymarchResult landscapeResult;
    landscapeResult.dist = 100000.0;
    landscapeResult.uvw = pos.xzy;
    landscapeResult.objectId = MAT_DEFAULT;
    
    
    vec3 landscapeDomain = pos + vec3(30.3,0,iTime);
    
    float scale = 10.0;
    
    float circleDist = CircleBomb(landscapeDomain.xz / scale, 2.0) * scale;

    circleDist -= CircleBomb(landscapeDomain.xz * 3.0, 1.0) * 0.3;

    float bump = SmoothNoise(landscapeDomain.xz * 5.0) * 0.1;

    float sliceCount = 2.0;

    for ( float slice = 0.0; slice >= -(sliceCount-1.0); slice -= 1.0 )
    {
        float sliceDist = circleDist + (slice) * .1 * scale - 1.0;    
            
        float sliceY = -slice * slice * sliceHeight + sliceBegin - bump; 
    
        sliceDist = -sminCubic( -sliceDist, (sliceY - landscapeDomain.y), 0.5 );
            
        landscapeResult.dist = sminCubic( landscapeResult.dist, sliceDist, 0.5 );
    }
    
    landscapeResult.dist = sminCubic( landscapeResult.dist, pos.y - (-sliceCount * sliceHeight + sliceBegin) - bump, 0.5 );    
    
    RaymarchResult result = landscapeResult;


    return result;
}

SurfaceInfo Scene_GetSurfaceInfo( Ray ray, TraceResult traceResult )
{
    SurfaceInfo surfaceInfo;
    
    surfaceInfo.pos = traceResult.pos;
    surfaceInfo.normal = traceResult.normal;
    
    surfaceInfo.albedo = vec3(1.0);
    surfaceInfo.r0 = vec3( 0.02 );
    surfaceInfo.gloss = 0.0;
    surfaceInfo.emissive = vec3( 0.0 );
        
    if ( traceResult.objectId == MAT_DEFAULT )
    {
        surfaceInfo.albedo = vec3(0.95, 0.95, 0.95); 
        surfaceInfo.gloss = 0.0;
        surfaceInfo.r0 = vec3( 0.02 );
    }
    
    if ( traceResult.objectId == MAT_QUAD )
    {
        surfaceInfo.albedo = vec3(0);
        surfaceInfo.gloss = 0.9;
        surfaceInfo.r0 = vec3( 0.02 );
        
        
        vec3 dir = refract( ray.dir, surfaceInfo.normal, 1.0 / 1.33 );
        
        vec3 boxRayDir = BoxDomainRotate( normalize( dir ) ) / (boxDimensions);
        vec3 boxRayPos = traceResult.uvw / (boxDimensions);

        vec3 h = -(boxRayPos - sign(boxRayDir)) / boxRayDir;
        float t = min(min(h.x, h.y), h.z);
        
        vec3 p = boxRayPos + boxRayDir * t;


        vec2 uv = p.xy;
        float d = length( boxRayDir * t * boxDimensions );
        
        uv = uv * 0.5 + 0.5;
        
        ivec2 tSize = textureSize(iChannel0, 0);
        uv = uv - 0.5;
        if ( tSize.x > tSize.y )
        {
            uv.x *= float(tSize.y) / float(tSize.x);
        }
        else
        {
            uv.y *= float(tSize.x) / float(tSize.y);
        }
        uv = uv + 0.5;
        
        vec3 emissiveSample = texture( iChannel0, uv ).rgb;
        surfaceInfo.emissive = emissiveSample * emissiveSample;

        surfaceInfo.emissive *= exp( -d * (1.0 - glassColour) * glassDensity );                      
    }   
    
    return surfaceInfo;    
}

vec3 Env_GetSkyColor( Ray ray )
{
    return vec3(0.0);
}

SurfaceLighting Scene_GetSurfaceLighting( Ray ray, SurfaceInfo surfaceInfo )
{   
    SurfaceLighting surfaceLighting;
    
    surfaceLighting.diffuse = vec3(0.0);
    surfaceLighting.specular = vec3(0.0);    
    
    Light_AddDirectional( surfaceLighting, surfaceInfo, ray.dir, sunDir, sunColor );
    
    
    Ray aoRay;
    aoRay.pos = surfaceInfo.pos;
    aoRay.dir = surfaceInfo.normal;
    float fAO = Scene_GetAmbientOcclusion(aoRay);    
    surfaceLighting.diffuse += fAO * (surfaceInfo.normal.y * 0.5 + 0.5) * ambientColor;
    
    return surfaceLighting;
}

float Env_GetFogFactor( Ray ray, float dist )
{    

    float fogAmount = fogDensity * exp(-ray.pos.y*fogHeightFalloff) * (1.0-exp(-dist*ray.dir.y*fogHeightFalloff ))/ray.dir.y;
    
    return exp(dist * -fogAmount);      
}

vec3 Env_GetFogColour(Ray ray)
{    
    return vec3(0.1, 0.35, 0.9);
}

vec3 Env_ApplyAtmosphere( vec3 colour, Ray ray, float dist )
{
    vec3 result = colour;
        
    float fogFactor = Env_GetFogFactor( ray, dist );
    vec3 fogColor = Env_GetFogColour( ray );
    result = mix( fogColor, result, fogFactor );

    return result;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 coord = uv - 0.5;
    coord.x *= iResolution.x / iResolution.y;
    
    CameraState camera;

    camera.pos = vec3(0,-0.3,-3);
    camera.target = vec3(0,0.1,0);
    camera.up = vec3(0,1,0);

    camera.pos = RotateY( camera.pos, sin(iTime * 0.1) * 0.3 );
    
    camera.pos *= (sin( iTime * 0.234 ) * 0.5 + 0.5) * 1.0 + 1.0;
    
    Ray ray = GetCameraRay( coord, camera );
    
    vec3 sceneColour = Scene_GetColour( ray );
    
    vec3 colour = sceneColour;
    
    colour = 1.0f - exp( -colour * 1.0 );
    
    colour = pow( colour, vec3(1.0f / 2.2f) );
    
    fragColor = vec4(colour, 1.0f);
}
