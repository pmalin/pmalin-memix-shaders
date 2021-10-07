//----------------------------------------------------------------------------------
// Fragment shader for the "Custom" filter (last in the catalog)
//----------------------------------------------------------------------------------
// Available uniforms:
//
// uniform float     iTime       : time in seconds
// uniform int       iFrame      : frame since the filter started running
// uniform vec3      iResolution : viewport resultion in .xy, horizontal crop in .z 
// uniform float     iParam[2]   : UI slider positions (from 0.0 to 1.0)
// uniform sampler2D iChannel0   : the webcam image
// uniform sampler2D iChannel1   : 256x256 grayscale random numbers
//----------------------------------------------------------------------------------



// This function computes the UV coords needed for fetching colors from iChannel0.
// Give it a pair of normalized coordinates in 0.0 to 1.0 and get the UVs you need
// to use in your texture() call for the shader to work well with different cams.
//
vec2 getuv( in vec2 p ) { return vec2( 0.5+(p.x-0.5)*iResolution.z, 1.0-p.y ); }

// This is an example of how to create a matte mask to do green screen composition
// Pass the webcam image as received from iChannel0, and this function will return
// a mask to segmentate foreground from background in "k" (0.0 to 1.0) and it will
// also correct the color of the foreground a little to compensate for the loss in
// green. Adapted from user Casty in Shadertoy's comments to the following shader:
// https://www.shadertoy.com/view/XsfGzn)
//
float getChromaKey( inout vec3 col )
{
    float maxrb = max( col.r, col.b );
    float k = clamp( (col.g-maxrb)*10.0, 0.0, 1.0 );
    float dg = col.g; 
    col.g = min( col.g, maxrb*0.8 ); 
    col += dg - col.g;
    return k;
}



// "Roto-twist" by iq (Inigo Quilez)
// https://www.shadertoy.com/view/XdfGzn
//
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.xy;

    float speed = 0.5 + iParam[1];
    float r = length(p);
    float an = 0.3*sin(iTime*speed*0.5 + r*5.0 );

    vec2 cst = vec2( cos(an), sin(an) );
    mat2 tra = mat2(cst.x,-cst.y,cst.y,cst.x)*(0.9+(iParam[1]-0.5));

    vec3 col = texture(iChannel0,getuv(0.5+0.5*tra*p)).xyz;
    fragColor = vec4( col, 1.0 );
}