// ZX Spectrum SCREEN$ - Memix
// by @P_Malin
//
// https://www.shadertoy.com/view/7scXRX
// This is a single pass version of:
//    https://www.shadertoy.com/view/ss3Xzj
//
// A silly shader designed to work with Memix https://www.memix.app/
// Save the shader as %LOCALAPPDATA%\Beautypi\Memix\shader.txt
//
// https://github.com/pmalin/pmalin-memix-shaders
//
// Thanks to Dave Hoskins for the hash function from https://www.shadertoy.com/view/4djSRW
//
// Another Sinclair ZX Spectrum Filter!
// Probably a lot slower than others as this does a more exhaustive search to find which attributes to use for each character cell. Also has some hacky CRT effects.
// Try changing the iChannel0 input.

// There is a slightly cheaper version of the colour selection code if you `#define SLOW_SEARCH 0`

#define MEMIX_FLIP_Y 1
#define APPLY_FADE 0

// Slower search producing slightly different colours
#define SLOW_SEARCH 1

#define APPLY_SCANLINES 1
#define APPLY_INTERFERENCE 1
#define APPLY_INTERFERENCE_JITTER 1

vec2 resolution=vec2(256, 192);
vec2 blockSize=vec2(8);

float intensity0 = 0.85;
float intensity1 = 1.0;

uniform float Param0;

struct BlockColours
{
    vec3 colourA;
    vec3 colourB;
};



mat4 bayerMatrix = mat4(
    vec4( 0.0/16.0, 12.0/16.0,  3.0/16.0, 15.0/16.0),
    vec4( 8.0/16.0,  4.0/16.0, 11.0/16.0,  7.0/16.0),
    vec4( 2.0/16.0, 14.0/16.0,  1.0/16.0, 13.0/16.0),
    vec4(10.0/16.0, 06.0/16.0,  9.0/16.0,  5.0/16.0));

float GetBayer( vec2 pixelCoord )
{
    return bayerMatrix[int(pixelCoord.x)%4][int(pixelCoord.y)%4];
}


float Dist2( vec3 a, vec3 b )
{
    vec3 ab = b-a;
    return dot(ab,ab);
}


vec3 RGBtoYUV(vec3 rgb)
{
    float y = dot(rgb, vec3(0.255, 0.587, 0.114));
    float u = 0.492 * (rgb.b - y);
    float v = 0.877 * (rgb.r - y);
    return vec3(y,u,v);
}

vec3 YUVtoRGB(vec3 yuv)
{
    float y = yuv.x;
    float u = yuv.y;
    float v = yuv.z;
    float r = y + 1.14 * v;
    float g = y - 0.395 * u - 0.581 * v;
    float b = y + 2.033 * u;
    return vec3(r,g,b);
}


vec3 SamplePixel( vec2 pixelCoord, sampler2D srcImage, vec2 srcResolution )
{
    vec2 uv = pixelCoord / resolution;
        
    float sourceAspect = srcResolution.x / srcResolution.y;
    float dstAspect = resolution.x / resolution.y;
    
    float aspectAdjust = dstAspect / sourceAspect;
    
    if ( aspectAdjust > 0.0 )
    {
        uv.x -= 0.5;
        uv.x *= aspectAdjust;
        uv.x += 0.5;
    }
    else
    {
        uv.y -= 0.5;
        uv.y /= aspectAdjust;
        uv.y += 0.5;    
    }
    
    if ( uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0 )
    {
        return vec3(intensity0);
    }
       
    #if MEMIX_FLIP_Y
    uv.y = 1.0 - uv.y;
    #endif

    return texture(srcImage, uv).rgb;
}

float GetDitherIntensity()
{
    float dither = 0.3;
    
    //dither = iMouse.x / iResolution.x;

    return dither;
}

vec3 ApplyDither( vec3 col, float value )
{
    float intensity = GetDitherIntensity();
    
    //col =  col * (1.0 - intensity) + intensity * value;
    col =  col + intensity * value - intensity * 0.5;
    
    col = clamp( col, vec3(0), vec3(1) );
    
    return col;
}


#if SLOW_SEARCH

BlockColours GetBlockColours( vec2 pixelCoord, sampler2D srcImage, vec2 srcResolution )
{
    BlockColours result;
    
    float testIntensity = (intensity0 + intensity1) * 0.5;

    vec2 blockOrigin = floor( pixelCoord / blockSize ) * blockSize;
        
    float error[8*8];
    for(int i=0; i<8; i++)
    {
        for(int j=0; j<8; j++)
        {
            if (i>j)
            {
                int index = i+j*8;
                error[index] = 0.0;
            }
        }
    }
        
    for( int y=0; y<int(blockSize.y); y++ )
    {
        for( int x=0; x<int(blockSize.x); x++ )
        {
            vec2 samplePixelCoord = blockOrigin + vec2(x,y);
            vec3 pixelColour = SamplePixel( samplePixelCoord, srcImage, srcResolution );
            
            float hash = GetBayer( samplePixelCoord );
            pixelColour = ApplyDither(pixelColour, hash);
                        
            for(int i=0; i<8; i++)
            {
                vec3 testColA = vec3(0);
                if ( (i & 1) != 0 ) testColA.r = testIntensity;
                if ( (i & 2) != 0 ) testColA.g = testIntensity;
                if ( (i & 4) != 0 ) testColA.b = testIntensity; 
                                
                for(int j=0; j<8; j++)
                {
                    if (i>j) 
                    {
                        vec3 testColB = vec3(0);

                        if ( (j & 1) != 0 ) testColB.r = testIntensity;
                        if ( (j & 2) != 0 ) testColB.g = testIntensity;
                        if ( (j & 4) != 0 ) testColB.b = testIntensity;

                        float dist = 0.0;
                        int bright = 0;

                        float dist1 = Dist2( testColA, pixelColour );
                        dist = dist1;
                        
                        float dist2 = Dist2( testColB, pixelColour );
                        if ( dist2 < dist1 )
                        {
                            dist = dist2;
                        }

                        error[i+j*8] += dist;
                    }
                }
            }
        }
    }        

    float smallestError = 999999.0;
    int smallestErrorIndex = 0;
    for(int i=0; i<8; i++)
    {
        for(int j=0; j<8; j++)
        {
            if (i>j)
            {
                int index = i+j*8;
                if ( error[index] < smallestError )
                {
                    smallestError = error[index];
                    smallestErrorIndex = index;
                }
            }
        }
    }
        
    result.colourA = vec3(0);    
    if ( (smallestErrorIndex & 1) != 0 ) result.colourA.r = 1.0;
    if ( (smallestErrorIndex & 2) != 0 ) result.colourA.g = 1.0;
    if ( (smallestErrorIndex & 4) != 0 ) result.colourA.b = 1.0;


    result.colourB = vec3(0);
    if ( (smallestErrorIndex & 8) != 0 ) result.colourB.r = 1.0;
    if ( (smallestErrorIndex & 16) != 0 ) result.colourB.g = 1.0;
    if ( (smallestErrorIndex & 32) != 0 ) result.colourB.b = 1.0;
    
    // determine if we should use "bright" attribute
    float scoreA = 0.0;
    float scoreB = 0.0;
    
    for( int y=0; y<int(blockSize.y); y++ )
    {
        for( int x=0; x<int(blockSize.x); x++ )
        {
            vec2 samplePixelCoord = blockOrigin + vec2(x,y);
            vec3 samp = SamplePixel( samplePixelCoord, srcImage, srcResolution );
            
            float d1 = Dist2(result.colourA * intensity0, samp);
            float d2 = Dist2(result.colourB * intensity0, samp);
            if (d1 < d2) scoreA += d1; else scoreA += d2;
            
            float d3 = Dist2(result.colourA * intensity1, samp);
            float d4 = Dist2(result.colourB * intensity1, samp);
            if (d3 < d4) scoreB += d3; else scoreB += d4;            
        }
    }        
    
    float intensity = intensity0;
    if (scoreA>scoreB) intensity = intensity1; 
    
    result.colourA *= intensity;
    result.colourB *= intensity;

    return result;    
}

#else 

// faster but lower quality
BlockColours GetBlockColours( vec2 pixelCoord, sampler2D srcImage, vec2 srcResolution )
{
    BlockColours result;
    
    float intensity0 = 0.85;
    float intensity1 = 1.0;
    
    float testIntensity = (intensity0 + intensity1) * 0.5;
    
    vec2 blockOrigin = floor( pixelCoord / blockSize ) * blockSize;
    
    float freq[8];
    for(int i=0; i<8; i++)
        freq[i] = 0.0;
        
    for( int y=0; y<int(blockSize.y); y++ )
    {
        for( int x=0; x<int(blockSize.x); x++ )
        {
            vec2 samplePixelCoord = blockOrigin + vec2(x,y);
            float hash = GetBayer( samplePixelCoord );
            vec3 pixelColour = SamplePixel( samplePixelCoord, srcImage, srcResolution );
                        
            pixelColour = ApplyDither( pixelColour, hash);
            
            
            int closestIndex = 0;
            float closestDist = 999.0;
            
            for(int i=0; i<8; i++)
            {
                vec3 testCol = vec3(0);
                
                if ( (i & 1) != 0 ) testCol.r = testIntensity;
                if ( (i & 2) != 0 ) testCol.g = testIntensity;
                if ( (i & 4) != 0 ) testCol.b = testIntensity;                
                
                float dist = Dist2( testCol, pixelColour );
                
                if ( dist < closestDist )
                {
                    closestIndex = i;
                    closestDist = dist;
                }
            }
            
            freq[closestIndex] += 1.0;
        }
    }
    
    float highestFreq = -1.0;
    int highestIndex = 0;
    
    for( int i=0; i<8; i++ )
    {
        if ( freq[i] > highestFreq )
        {
            highestFreq = freq[i];
            highestIndex = i;
            
        }
    }

    for(int i=0; i<8; i++)
        freq[i] = 0.0;
        
    for( int y=0; y<int(blockSize.y); y++ )
    {
        for( int x=0; x<int(blockSize.x); x++ )
        {
            vec2 samplePixelCoord = blockOrigin + vec2(x,y);
            float hash = GetBayer( samplePixelCoord );
            vec3 pixelColour = SamplePixel( samplePixelCoord, srcImage, srcResolution );
            
            pixelColour = ApplyDither(pixelColour, hash);
            
            int closestIndex = 0;
            float closestDist = 999.0;
            
            for(int i=0; i<8; i++)
            {
                {
                    vec3 testCol = vec3(0);

                    if ( (i & 1) != 0 ) testCol.r = testIntensity;
                    if ( (i & 2) != 0 ) testCol.g = testIntensity;
                    if ( (i & 4) != 0 ) testCol.b = testIntensity;                

                    float dist = length( testCol - pixelColour );

                    if ( dist < closestDist )
                    {
                        closestIndex = i;
                        closestDist = dist;
                    }
                }
            }

            // ignore pixels that will map to highestIndex
            if( closestIndex != highestIndex )
            {            
                freq[closestIndex] += 1.0;
            }
        }
    }
    
    float highestFreq2 = -1.0;
    int highestIndex2 = 7;
    
    for( int i=0; i<8; i++ )
    {
        if ( i != highestIndex )
        {
            if ( freq[i] > highestFreq2 )
            {
                highestFreq2 = freq[i];
                highestIndex2 = i;

            }
        }
    }
    
    result.colourA = vec3(0);    
    if ( (highestIndex & 1) != 0 ) result.colourA.r = 1.0;
    if ( (highestIndex & 2) != 0 ) result.colourA.g = 1.0;
    if ( (highestIndex & 4) != 0 ) result.colourA.b = 1.0;

    result.colourB = vec3(0);
    if ( (highestIndex2 & 1) != 0 ) result.colourB.r = 1.0;
    if ( (highestIndex2 & 2) != 0 ) result.colourB.g = 1.0;
    if ( (highestIndex2 & 4) != 0 ) result.colourB.b = 1.0;    


    // determine if we should use "bright" attribute
    float scoreA = 0.0;
    float scoreB = 0.0;
    
    for( int y=0; y<int(blockSize.y); y++ )
    {
        for( int x=0; x<int(blockSize.x); x++ )
        {
            vec2 samplePixelCoord = blockOrigin + vec2(x,y);
            vec3 samp = SamplePixel( samplePixelCoord, srcImage, srcResolution );
            
            float d1 = length(result.colourA * intensity0 - samp);
            float d2 = length(result.colourB * intensity0 - samp);
            if (d1 < d2) scoreA += d1; else scoreA += d2;
            
            float d3 = length(result.colourA * intensity1 - samp);
            float d4 = length(result.colourB * intensity1 - samp);
            if (d3 < d4) scoreB += d3; else scoreB += d4;            
        }
    }    

    float intensity = intensity0;
    if (scoreB < scoreA) intensity = intensity1;

    
    result.colourA *= intensity;
    result.colourB *= intensity;
    
    return result;
}

#endif


vec3 GetPixelColour( BlockColours blockColours, vec2 pixelCoord, sampler2D srcImage, vec2 srcResolution, int loadingByteIndex )
{        
    vec3 pixelColour = SamplePixel(pixelCoord, srcImage, srcResolution );
    float hash = GetBayer( pixelCoord );
    pixelColour = ApplyDither( pixelColour, hash );    

    vec3 colA = blockColours.colourA;
    vec3 colB = blockColours.colourB;

    int attributeByteIndex = (256 * 192/8) + int( floor(pixelCoord.x / 8.) + (24.-floor(pixelCoord.y / 8.)) * (resolution.x / 8.) );
    
    if ( attributeByteIndex > loadingByteIndex )
    {
        colA = vec3(0);
        colB = vec3(1) * intensity0;
    }
    
    int pixelY = int(resolution.y) - 1 - int(pixelCoord.y);
    int pixelYBlock = pixelY / (8*8);
    int pixelYChar = (pixelY & (8*8-1)) / 8;
    int pixelYCharRow = (pixelY & (8*8-1)) % 8;
    int pixelYIndex = pixelYChar + pixelYCharRow * 8 + pixelYBlock * 8*8;
    
    int pixelByteIndex = int(pixelCoord.x) / 8 + pixelYIndex * int(resolution.x)/8;

    if (pixelByteIndex > loadingByteIndex)
    {
        pixelColour = vec3(1);
    }

    float dist1 = Dist2( colA, pixelColour );
    float dist2 = Dist2( colB, pixelColour );
    if ( dist1 < dist2 )
    {
        return colA;
    }
    else
    {                   
        return colB;
    }                        
}

vec4 hash41(float p)
{
    // From: Hash without Sine by Dave Hoskins
    // https://www.shadertoy.com/view/4djSRW

    vec4 p4 = fract(vec4(p) * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);    
}

vec4 InterferenceSmoothNoise1D( float x )
{
    float f0 = floor(x);
    float fr = fract(x);

    vec4 h0 = hash41( f0 );
    vec4 h1 = hash41( f0 + 1.0 );

    return h1 * fr + h0 * (1.0 - fr);
}


vec4 InterferenceNoise( vec2 uv )
{
    float scanLine = floor(uv.y * resolution.y); 
    float scanPos = scanLine + uv.x;
    float timeSeed = fract( iTime * 123.78 );
    
    return InterferenceSmoothNoise1D( scanPos * 234.5 + timeSeed * 12345.6 );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    int loadingByteIndex = (int(iTime * 60.0) * 3);

    int loadingByteEnd = (192*256/8+(32*24) + 64);


#if APPLY_FADE    
    float fade = float(loadingByteIndex - loadingByteEnd - 512) / 400.0;
    fade = clamp( fade, 0.0, 1.0 );
#else
    float fade = 0.0;    
#endif
    
    vec2 uv = fragCoord/iResolution.xy;
        
    vec3 col = vec3(1);
    
    ivec2 tSize = textureSize(iChannel0, 0);
    vec2 inputResolution = vec2(tSize);
    
    uv = uv - 0.5;
    
    float outputAspect = iResolution.x / iResolution.y;
    float screenAspect = (resolution.x / resolution.y);
    float aspectAdjust = outputAspect / screenAspect;
    
    if ( aspectAdjust > 1.0 )
    {
        uv.x *= aspectAdjust;
    }
    else
    {
        uv.y /= aspectAdjust;
    }
    
    uv *= mix( 1.1, 1.0, fade );
    uv = uv + 0.5;
    
    vec4 noise = InterferenceNoise( uv );
    noise = mix( noise, vec4(0), fade );

#if APPLY_INTERFERENCE_JITTER
    uv.x += noise.w * 0.0015;
#endif    
    
    vec2 pixelCoord = floor(uv * resolution);

    BlockColours blockColours = GetBlockColours( pixelCoord, iChannel0, inputResolution );

    col = GetPixelColour( blockColours, pixelCoord, iChannel0, inputResolution, loadingByteIndex );
    
    {
        vec3 origCol = SamplePixel(uv * resolution, iChannel0, inputResolution ).rgb;
        col = mix( col, origCol, fade );
    }

    if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0)
    {
        //col = vec3(intensity0) * vec3(0,1,1);
        col = vec3(intensity0) * vec3(0,0,0);
        
        if ( loadingByteIndex < loadingByteEnd )
        {
            float t =(uv.y*300.0+uv.x)* 0.4+iTime * 3000.0;
            float raster = t / 150.0;
            
            float barSize = 25.0;
            float scrollSpeed = 1.0;
            float blend = step(fract(raster * barSize + iTime * scrollSpeed + sin(iTime * 20.0 + raster * 16.0)), 0.5);            
            col = mix( vec3(0,0,1), vec3(1,1,0), blend ) * intensity0;
        }
    }

#if APPLY_INTERFERENCE    
    col += (noise.xxx * 1.0 - 0.5) * 0.05;
#endif  

#if APPLY_SCANLINES
    float scanline = 1.0;
    /*{
        float a = resolution.y * 3.14 * 2.0;
        float b = - 3.14 * 0.5;
        scanline = sin(uv.y * a + b);
    }*/
    {
        float a = resolution.y * 3.14 * 2.0;
        float b = - 3.14 * 0.5;
        float x = uv.y;

        float delta = length(vec2(dFdx(x),dFdy(x)));
        float v1 = -cos(a*(x-delta)+b)/a;
        float v2 = -cos(a*(x+delta)+b)/a;
        scanline = (v2 - v1) / delta;
    }
    
    scanline = scanline * 0.5 + 0.5;
    float scanlineIntensity = 0.4;
    scanlineIntensity = mix( scanlineIntensity, 0.0, fade );
    scanline = mix( 1.0, scanline, scanlineIntensity );
    col = col * col;
    float ambient = 0.01;
    col = mix(col, vec3(1.0), ambient);
    col = col * scanline;        
    col = sqrt(col);
#endif   

    //if ( uv.x > 0.5 ) col = currPixelColour;

    // Output to screen
    fragColor = vec4(col,1.0);    
    
}