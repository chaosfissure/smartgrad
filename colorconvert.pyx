LAB_MIN    = (  0.0, -86.9849162448182,  -118.79680598946653)
LAB_MAX    = (100.0,  97.30703006302993,   88.72927586118658)
LAB_SPAN   = tuple(large-small for large, small in zip(LAB_MAX, LAB_MIN))


def ToSRGB(what):
	return tuple(max(min(255, int(255*x)), 0) for x in what)

def FromSRGB(what):
	return tuple(x/255.0 for x in what)
	
def QuantizeLAB(lab):
	l = (lab[0] - LAB_MIN[0]) / LAB_SPAN[0]
	a = (lab[1] - LAB_MIN[1]) / LAB_SPAN[1]
	b = (lab[2] - LAB_MIN[2]) / LAB_SPAN[2]
	return l, a, b
	
	
cdef extern from "math.h":
	cdef double pow(double, double)
	
	
cdef unsigned RGBToInt(unsigned char r, unsigned char g, unsigned char b):
	'''
	Encode RGB value into a integer pixel value.
	Technically, it gets stored as BBGGRR.
	'''
	return r | (g<<8) | (b<<16)
	
	
def IntToRGB(unsigned rgb):
	'''
	Unpack BBGGRR into r, g, b
	'''
	cdef unsigned r = (rgb      ) & 255
	cdef unsigned g = (rgb >>  8) & 255
	cdef unsigned b = (rgb >> 16) & 255
	return r, g, b
	
	

cdef double RGB_XYZ_Clamp(double val):
	if val > 0.04045:
		return pow(((val + 0.055) / 1.055), 2.4)
	return val / 12.92

	
cdef color RGB_To_XYZ(color tmp):
	cdef double x = RGB_XYZ_Clamp(tmp.x) * 100.0
	cdef double y = RGB_XYZ_Clamp(tmp.y) * 100.0
	cdef double z = RGB_XYZ_Clamp(tmp.z) * 100.0
	
	cdef color result
	result.x = x*0.4124 + y*0.3576 + z*0.1805
	result.y = x*0.2126 + y*0.7152 + z*0.0722
	result.z = x*0.0193 + y*0.1192 + z*0.9505

	return result
	

cdef double XYZ_RGB_Clamp(double val):
	if val > 0.0031308:
		return 1.055 * pow(val, 1.0/2.4) - 0.055
	return val * 12.92

	
cdef color XYZ_To_RGB(color what):
	cdef double x = what.x / 100.0
	cdef double y = what.y / 100.0
	cdef double z = what.z / 100.0
	
	cdef color result
	result.x = XYZ_RGB_Clamp(x* 3.2406 + y*-1.5372 + z*-0.4986)
	result.y = XYZ_RGB_Clamp(x*-0.9689 + y* 1.8758 + z* 0.0415)
	result.z = XYZ_RGB_Clamp(x* 0.0557 + y*-0.2040 + z* 1.0570)
	return result
	

cdef double XYZ_LAB_Clamp(double val):
	if val > 0.008856:
		return pow(val, 1.0/3.0)
	return (7.787 * val) + (16.0 / 116.0)


cdef color XYZ_To_LAB(color what):

	# D55_LIGHT = [95.682, 100.000, 92.149]
	cdef double x = XYZ_LAB_Clamp(what.x /  95.682)
	cdef double y = XYZ_LAB_Clamp(what.y / 100.000)
	cdef double z = XYZ_LAB_Clamp(what.z /  92.149)
	
	cdef color result
	result.x = (y*116.0) - 16.0
	result.y = (x - y) * 500.0
	result.z = (y - z) * 200.0

	return result
	       			
			
cdef double LAB_XYZ_Clamp(double val):
	# Solved min value for x^3 > 0.008856
	if val >= 0.206893:
		return val*val*val
		
	return (val - 16.0 / 116.0) / 7.787


cdef color LAB_To_XYZ(color what):
	cdef color result
	result.y = (what.x + 16.0) / 116.0
	result.x = (what.y / 500.0) + result.y
	result.z = result.y - (what.z / 200.0)
	
	# D55_LIGHT = [95.682, 100.000, 92.149]
	result.x = LAB_XYZ_Clamp(result.x) *  95.682
	result.y = LAB_XYZ_Clamp(result.y) * 100.000
	result.z = LAB_XYZ_Clamp(result.z) *  92.149
	
	return result
	
	
def RGB_To_LAB(what):
	cdef color color_in
	color_in.x = what[0]
	color_in.y = what[1]
	color_in.z = what[2]
	
	cdef color color_out = RGB_To_XYZ(XYZ_To_LAB(color_in))
	return color_out.x, color_out.y, color_out.z

def LAB_To_RGB(what):
	cdef color color_in
	color_in.x = what[0]
	color_in.y = what[1]
	color_in.z = what[2]
	
	cdef color color_out = LAB_To_XYZ(XYZ_To_RGB(color_in))
	return color_out.x, color_out.y, color_out.z


