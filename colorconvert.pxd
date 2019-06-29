cdef struct color:
	double x
	double y
	double z

cdef color LAB_To_XYZ(color what)
cdef double LAB_XYZ_Clamp(double val)

cdef color XYZ_To_LAB(color what)
cdef double XYZ_LAB_Clamp(double val)

cdef color XYZ_To_RGB(color what)
cdef double XYZ_RGB_Clamp(double val)

cdef color RGB_To_XYZ(color tmp)
cdef double RGB_XYZ_Clamp(double val)

cdef unsigned RGBToInt(unsigned char r, unsigned char g, unsigned char b)