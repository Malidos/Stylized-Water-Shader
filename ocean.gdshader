shader_type spatial;
render_mode shadows_disabled;

#define CAUSTICS
#define FRESNEL
#define PLAYER_WAVES
#define DISPLACEMENT
#define SSR

group_uniforms color;
uniform vec3 absorption_color : source_color = vec3(1.0, 0.35, 0.0);
#ifdef FRESNEL
uniform float fresnel_radius : hint_range(0.0, 6.0, 0.01) = 2.0;
uniform vec3 fresnel_color : source_color = vec3(0.0, 0.57, 0.72);
#endif
uniform float roughness : hint_range(0.0, 1.0, 0.01) = 0.15;
uniform float specular : hint_range(0.0, 1.0, 0.01) = 0.25;
// Depth adjustment
uniform float depth_distance : hint_range(0.0, 50.0, 0.1) = 25.0;
uniform float beers_law : hint_range(0.0, 20.0, 0.1) = 4.5;

#ifdef DISPLACEMENT
group_uniforms displacement;
uniform float displacement_strength : hint_range(0.0, 5.0, 0.1) = 0.3;
uniform float displacement_scroll_speed : hint_range(0.0, 1.0, 0.001) = 0.1;
uniform vec2 displacement_scroll_offset = vec2 (-0.2, 0.3);
uniform float displacement_scale_offset = 0.5;
uniform vec2 displacement_scale = vec2(0.04);
uniform sampler2D displacement_texture : hint_default_black, repeat_enable;
#endif

group_uniforms edge;
uniform float edge_thickness : hint_range(0.0, 1.0, 0.001) = 0.3;
uniform float edge_speed : hint_range(0.0, 1.0, 0.001) = 0.35;
uniform vec2 edge_noise_scale = vec2(0.4);
uniform sampler2D edge_noise : repeat_enable;
uniform sampler2D edge_ramp : repeat_disable;

#ifdef PLAYER_WAVES
group_uniforms player;
uniform float influence_size : hint_range(0.0, 4.0, 0.1) = 1.0;
uniform float player_wave_frequenzy : hint_range(0.0, 20.0, 0.1) = 10.0;
uniform float player_wave_speed : hint_range(0.0, 10.0, 0.1) = 5.0;
#endif

#ifdef CAUSTICS
group_uniforms caustics;
uniform float caustic_size : hint_range(0.0, 8.0, 0.01) = 2.0;
uniform float caustic_range : hint_range(0.0, 256.0, 0.1) = 40.0;
uniform float caustic_strength : hint_range(0.0, 1.0, 0.01) = 0.08;
#endif

#ifdef SSR
group_uniforms screen_space_reflections;
uniform float ssr_mix_strength : hint_range(0.0, 1.0, 0.01) = 0.65;
uniform float ssr_travel : hint_range(0.0, 300.0, 0.5) = 100.0;
uniform float ssr_resolution_near : hint_range(0.1, 10.0, 0.1) = 1.0;
uniform float ssr_resolution_far : hint_range(2.0, 20.0, 0.1) = 5.0;
uniform float ssr_tolerance : hint_range(0.0, 2.0, 0.01) = 1.0;
#endif

group_uniforms normal_map;
uniform float refraction_strength : hint_range(0.0, 4.0, 0.01) = 1.25;
uniform float normal_map_strength : hint_range(0.0, 4.0, 0.01) = 1.0;
uniform float scroll_speed : hint_range(0.0, 1.0, 0.01) = 0.3;
uniform vec2 scroll_offset = vec2(0.1, -0.3);
uniform float scale_offset = 0.5;
uniform vec2 normal_map_scale = vec2(0.1);
uniform sampler2D normal_map : hint_normal, filter_linear_mipmap;

// Hidden Uniforms
global uniform float wind_intensity; // Global shader parameter between 0.0 and 1.0
global uniform vec3 wind_direction;
#ifdef PLAYER_WAVES
global uniform vec3 player_position;
#endif
uniform sampler2D screen_texture: hint_screen_texture, filter_linear_mipmap, repeat_disable;
uniform sampler2D depth_texture: hint_depth_texture, filter_linear_mipmap, repeat_disable;

varying vec3 global_position;

#ifdef CAUSTICS
// Permutation polynomial hash credit Stefan Gustavson
vec4 permute(vec4 t) {
    return t * (t * 34.0 + 133.0);
}

// Gradient set is a normalized expanded rhombic dodecahedron
vec3 grad(float hash) {

    // Random vertex of a cube, +/- 1 each
    vec3 cube = mod(floor(hash / vec3(1.0, 2.0, 4.0)), 2.0) * 2.0 - 1.0;

    // Random edge of the three edges connected to that vertex
    // Also a cuboctahedral vertex
    // And corresponds to the face of its dual, the rhombic dodecahedron
    vec3 cuboct = cube;
    cuboct[int(hash / 16.0)] = 0.0;

    // In a funky way, pick one of the four points on the rhombic face
    float type = mod(floor(hash / 8.0), 2.0);
    vec3 rhomb = (1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));

    // Expand it so that the new edges are the same length
    // as the existing ones
    vec3 grad = fma(cuboct, vec3(1.22474487139), rhomb);

    // To make all gradients the same length, we only need to shorten the
    // second type of vector. We also put in the whole noise scale constant.
    // The compiler should reduce it into the existing floats. I think.
    grad *= fma(-0.042942436724648037, type, 1.0) * 3.5946317686139184;

    return grad;
}

// BCC lattice split up into 2 cube lattices
vec4 os2NoiseWithDerivativesPart(vec3 X) {
    vec3 b = floor(X);
    vec4 i4 = vec4(X - b, 2.5);

    // Pick between each pair of oppposite corners in the cube.
    vec3 v1 = b + floor(dot(i4, vec4(.25)));
    vec3 v2 = b + vec3(1, 0, 0) + vec3(-1, 1, 1) * floor(dot(i4, vec4(-.25, .25, .25, .35)));
    vec3 v3 = b + vec3(0, 1, 0) + vec3(1, -1, 1) * floor(dot(i4, vec4(.25, -.25, .25, .35)));
    vec3 v4 = b + vec3(0, 0, 1) + vec3(1, 1, -1) * floor(dot(i4, vec4(.25, .25, -.25, .35)));

    // Gradient hashes for the four vertices in this half-lattice.
    vec4 hashes = permute(mod(vec4(v1.x, v2.x, v3.x, v4.x), 289.0));
    hashes = permute(mod(hashes + vec4(v1.y, v2.y, v3.y, v4.y), 289.0));
    hashes = mod(permute(mod(hashes + vec4(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);

    // Gradient extrapolations & kernel function
    vec3 d1 = X - v1; vec3 d2 = X - v2; vec3 d3 = X - v3; vec3 d4 = X - v4;
    vec4 a = max(0.75 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a; vec4 aaaa = aa * aa;
    vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y);
    vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));

    // Derivatives of the noise
    vec4 derivative = -8.0 * mat4(vec4(d1,0.), vec4(d2,0.), vec4(d3,0.), vec4(d4,0.)) * (aa * a * extrapolations)
        + mat4(vec4(g1, 0.), vec4(g2, 0.), vec4(g3, 0.), vec4(g4, 0.)) * aaaa;

    // Return it all as a vec4
    return vec4(derivative.xyz, dot(aaaa, extrapolations));
}

// Rotates domain, but preserve shape. Hides grid better in cardinal slices.
// Good for texturing 3D objects with lots of flat parts along cardinal planes.
vec4 os2NoiseWithDerivatives_Fallback(vec3 X) {
    X = dot(X, vec3(2.0/3.0)) - X;

    vec4 result = os2NoiseWithDerivativesPart(X) + os2NoiseWithDerivativesPart(X + 144.5);

    return vec4(dot(result.xyz, vec3(2.0/3.0)) - result.xyz, result.w);
}
#endif

#ifdef FRESNEL
float fresnel(vec3 normal, vec3 view) {
	return pow((1.0 - clamp(dot(normalize(normal), normalize(view)), 0.0, 1.0 )), fresnel_radius);
}
#endif


vec2 refract_uv(inout vec2 uv, vec3 normal, float depth){
	float strength1 = refraction_strength * depth;
	uv += fma(strength1, length(normal), strength1 * -1.2);
	return uv;
}

#ifdef SSR
vec2 get_uv_from_view_position(vec3 position_view_space, mat4 proj_m)
{
	vec4 position_clip_space = proj_m * vec4(position_view_space.xyz, 1.0);
	vec2 position_ndc = position_clip_space.xy / position_clip_space.w;
	return position_ndc.xy * 0.5 + 0.5;
}

vec3 get_view_position_from_uv(vec2 uv, float depth, mat4 inv_proj_m)
{
	vec4 position_ndc = vec4((uv * 2.0) - 1.0, depth, 1.0);
	vec4 view_position = inv_proj_m * position_ndc;
	return view_position.xyz /= view_position.w;
}
#endif

bool in_bounds(vec2 uv) {
	vec2 fruv = abs(floor(uv));
	return fruv.x + fruv.y < 0.1;
}


void vertex() {
	global_position = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;

	#ifdef DISPLACEMENT
	float time = TIME * displacement_scroll_speed * fma(wind_intensity, 0.7, 0.3);
	float displace1 = texture(displacement_texture, fma(global_position.xz, displacement_scale, time * -wind_direction.xz)).r;
	float displace2 = texture(displacement_texture, fma(global_position.xz, displacement_scale * displacement_scale_offset, time * (-wind_direction.xz + displacement_scroll_offset))).r;
	float displacement_mixed = mix(displace1, displace2, 0.4);
	float offset = fma(displacement_mixed, 2.0, -1.0) * displacement_strength;
	VERTEX.y += offset;
	global_position.y += offset;
	#endif
}


void fragment() {
	vec3 opposing_color = vec3(1.0) - absorption_color.rgb;
	vec3 normalized_wind_direction = normalize(wind_direction);
	float wind_intens_factor = fma(wind_intensity, 0.7, 0.3);
	#ifdef FRESNEL
	float fresnel_value = fresnel(NORMAL, VIEW);
	#endif

	float time_factor = TIME * scroll_speed * wind_intens_factor;
	vec3 n1 = textureLod(normal_map, fma(global_position.xz, normal_map_scale, time_factor * -normalized_wind_direction.xz), 2.0).xyz;
	vec3 n2 = textureLod(normal_map, fma(global_position.xz, normal_map_scale * scale_offset, time_factor * 0.8 * (-normalized_wind_direction.xz + scroll_offset)), 2.0).xyz;
	NORMAL_MAP = mix(n1, n2, 0.5);
	NORMAL_MAP_DEPTH = normal_map_strength;

	float depth_tex = texture(depth_texture, SCREEN_UV).r;

	vec3 ndc = vec3(fma(SCREEN_UV, vec2(2.0), vec2(-1.0)), depth_tex);
	vec4 world = INV_VIEW_MATRIX * INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
	world.y /= world.w;
	float vertey_y = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).y;
	float relative_depth = vertey_y - world.y;


	// Create Edge caused by other Objects
	float edge_blend = clamp(relative_depth / -edge_thickness + 1.0, 0.0, 1.0);
	vec2 edge_noise_uv = global_position.xz * edge_noise_scale * fma(normalized_wind_direction.xz, vec2(0.5), vec2(0.5));
	edge_noise_uv = fma(-normalized_wind_direction.xz * TIME * edge_speed, vec2(wind_intens_factor), edge_noise_uv);
	float edge_noise_sample = texture(edge_noise, edge_noise_uv).r;
	float edge_mask = normalize( texture(edge_ramp, vec2(edge_noise_sample * fma(edge_blend, -1., 1.))).r);

	// Create Ripples caused by player
	float player_effect_mask = 0.0;
	#ifdef PLAYER_WAVES
	vec3 player_relative = vec3(global_position - player_position);
	float player_height = smoothstep(1.0, 0.0, abs(player_relative.y));
	float player_position_factor = smoothstep(influence_size, 0.0, length(player_relative.xz));
	float player_waves = pow( fma( sin(fma(player_position_factor, player_wave_frequenzy, TIME * player_wave_speed)), 0.5, 0.5), 6.0);
	float wave_distort = texture( edge_ramp, vec2( player_waves * (edge_noise_sample + 0.2) * player_position_factor * player_height)).x;
	player_effect_mask = clamp(normalize( fma(wave_distort, -1.0, 0.4)), 0.0, 1.0);
	#endif

	// combine Edge Mask with Player Ripples
	float ripple_mask = clamp( fma( edge_mask, edge_blend, player_effect_mask), 0.0, 1.0);

	// Calculate Fragment Depth
	vec4 clip_pos = PROJECTION_MATRIX * vec4(VERTEX, 1.0);
	clip_pos.xyz /= clip_pos.w;
	DEPTH = clip_pos.z;
	// Refract UV
	vec2 refracted_uv = SCREEN_UV;
	refract_uv(refracted_uv, NORMAL_MAP, sqrt(DEPTH) * relative_depth);

	vec3 screen;
	float depth_blend;
	float refracted_depth_tex = texture(depth_texture, refracted_uv).x;
	ndc = vec3(fma(refracted_uv, vec2(2.0), vec2(-1.0)), refracted_depth_tex);
	world = INV_VIEW_MATRIX * INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
	world.xyz /= world.w;
	float depth_test = vertey_y - world.y;

	// Caustic Effects
	#ifdef CAUSTICS
	float range_mod = clamp((VERTEX.z + caustic_range) * 0.05, 0.0, 1.0);
	float caustic_value = 0.0;
	// Protect yourself from calculating Noise at runtime with this handy if statement!
	if (range_mod > 0.0) {
		vec3 X = vec3(world.xz * caustic_size, mod(TIME, 578.0) * 0.8660254037844386);
		vec4 noiseResult = os2NoiseWithDerivatives_Fallback(X);
		noiseResult = os2NoiseWithDerivatives_Fallback(X - noiseResult.xyz / 16.0);
		caustic_value = fma(noiseResult.w, 0.5, 0.5) * range_mod * range_mod;
	}
	#endif

	/*
	Sometimes the Water Refraction would cause the sampling of a screen position that is either
	outside the screen bounds or where another object is infront of the water.
	Switching back to the unrefracted SCREEN_UV fixes that.
	*/

	if (depth_test > -0.0001 && in_bounds(refracted_uv)) {
		screen = texture(screen_texture, refracted_uv).rgb * 0.9;
		depth_blend = clamp(depth_test / depth_distance, 0.0, 1.0);
		depth_blend = fma(exp(-depth_blend * beers_law), -1.0, 1.0);
	} else {
		screen = texture(screen_texture, SCREEN_UV).rgb * 0.9;
		depth_blend = clamp(relative_depth / depth_distance, 0.0, 1.0);
		depth_blend = fma(exp(-depth_blend * beers_law), -1.0, 1.0);
	}

	#ifdef SSR
	vec3 view_normal_map = mat3(VIEW_MATRIX) * (vec3(NORMAL_MAP.x, 0.0, NORMAL_MAP.y) * 2.0 - 1.0);
	vec3 combined_normal = normalize(view_normal_map * (NORMAL_MAP_DEPTH * 0.15) + NORMAL);
	vec3 reflacted_path = reflect(-VIEW, combined_normal);

	vec2 current_screen_pos = vec2(0.0);
	vec3 current_view_pos = VERTEX;
	vec3 sampled_color = vec3(-1.0);
	float current_stepD = 0.0;
	float current_depth = 0.0;
	float alpha_hit = 0.0;
	for(float i = 0.01; i < ssr_travel; i++) {
		current_stepD = mix(ssr_resolution_near, ssr_resolution_far,float(i) / float(ssr_travel));
		current_view_pos += reflacted_path * current_stepD;
		current_screen_pos = get_uv_from_view_position(current_view_pos, PROJECTION_MATRIX);
		if (!in_bounds(current_screen_pos)) {break;}
		current_depth = get_view_position_from_uv(current_screen_pos, texture(depth_texture, current_screen_pos).x, INV_PROJECTION_MATRIX).z - current_view_pos.z;

		if (current_depth > -0.0001 && current_depth <= ssr_tolerance * current_stepD) {
			sampled_color = textureLod(screen_texture, current_screen_pos, 0.5).rgb;
			vec2 ruv = 1.0 - abs(current_screen_pos * 2.0 - 1.0);
			ruv = pow(ruv, vec2(0.5));
			alpha_hit = clamp(min(ruv.x, ruv.y), 0.0, 1.0);
			break;
		}
		i += current_stepD;
	}
	#endif

	vec3 color = clamp(screen - absorption_color.rgb * depth_blend, vec3(0.0), vec3(1.0)); // Absorb Screen Color
	color = mix(color, opposing_color, depth_blend*depth_blend); // Apply depth color
	#ifdef FRESNEL
	color = mix(color, fresnel_color, fresnel_value); // Apply fresnel color
	#endif
	#ifdef CAUSTICS
	color = clamp(color + caustic_value * caustic_strength * (1.0 - depth_blend), vec3(0.0), vec3(1.0));
	#endif
	#ifdef SSR
	color = mix(color, sampled_color, alpha_hit * (1.0 - roughness) * ssr_mix_strength);
	#endif
	color = mix(color, vec3(0.98), ripple_mask); // Apply Ripples
	ALBEDO = color;
	ROUGHNESS = roughness;
	SPECULAR = specular;
}