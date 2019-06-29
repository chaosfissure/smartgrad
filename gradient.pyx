import multiprocessing, time
import numpy as np
import cv2

from collections  import namedtuple
from colorconvert import *
from colorconvert cimport RGBToInt
from commonutils  import *
from CVUtils      import *
from PIL          import Image

SAMPLESIZE = 256
XYZSTR     = 'xyz'

SlimColorData = namedtuple('SlimColorData', ('rgb', 'lab'))	
FakeArgs      = namedtuple('FakeArgs',      ('gw',  'gt'))
	
class Gradient(object):
	'''
	A gradient takes input objects (assumed to be ColorData types), and 
	slims them down into only RGB and LAB data.  It additionally stores
	a bunch of OpenCV datatypes used to make comparing similarity of
	gradients faster and use a lot less code.  It's not as customizable,
	but it's "Good Enough" to work, and the speed benefit is amazing.
	'''
	
	COLOR_THRESHOLD = 36
	
	@staticmethod
	def ReencodeTest(string):
		
		colors = []
		for line in string.split('\n'):
		
			line = line.strip()
		
			if 'index=' in line and 'color=' in line:
				try:
					idx, color = [x.split('=')[1] for x in line.split()]
					rgb = IntToRGB(int(color))
					
					rgbf = FromSRGB(rgb)
					labf = RGB_To_LAB(rgbf)
					lab  = QuantizeLAB(labf)
					
					colors.append(SlimColorData(rgb, lab))
					
				except Exception as e:
					print(e)
					pass
					
			elif 'title=' in line:
				start  = line.find('title=')
				quote1 = line.find('"', start)
				quote2 = line.find('"', quote1+1)
				name = line[quote1+1:quote2]
					
		return Gradient(name, colors, FakeArgs(0,0))
					

	@staticmethod
	def Sort(a, b):
		''' 
		Returns  1 if a is darker than b.
		Returns -1 if b is darker than a.
		Returns  0 if they seem similar enough.
		'''		
		
		if self.for_sort < other.for_sort: 
			return 1
			
		elif other.for_sort < self.for_sort:
			return -1
			
		else:
			return 0

	def __init__(self, descr, colors, args):

		self.text   = descr
		
		if len(colors) != SAMPLESIZE:
			raise ValueError(f'Number of color entries is {len(colors)}, not {SAMPLESIZE}.')
		
		self.colors = tuple(SlimColorData(x.rgb, x.lab) for x in colors)
		
		# Set up the opencv stuff for speed in comparing images. --------------
		self.img = Image.new('RGB', (len(self.colors), 1), 0x0)
		pixels   = self.img.load()
		for i, color in enumerate(self.colors):
			pixels[i,0] = color.rgb
				
		cv_img = PILToOpenCV(self.img)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
		
		self.for_sort = sum(color.lab[0] for color in self.colors)
			
		# Blur the image to smooth out neighbors when looking at regions around a pixel
		# for similarities.
		
		if args.gw > 1:	
			if args.gw % 2 == 0:
				args.gw -= 1 # gaussian blur has to have odd numbers as kernel dimensions
			cv_img = cv2.GaussianBlur(cv_img, (args.gw,1), 0)
			cv_img = cv2.medianBlur(cv_img, 3)
				
		self.gray = cv_img
		self.required_differences = args.gt * len(self.colors) // 100
				
	def GetImage(self):
		'''
		Although we store a "width x 1" image, the full sized image
		is created on demand since it would require substantially more
		memory.
		'''
		scale_w  = self.img.size[0]*3
		scale_h  = scale_w // 2
		return self.img.resize((scale_w, scale_h))		
		
	def IsDuplicate(self, other):
		'''
		Checks if the blurred gray version of the image is similar enough to
		the blurred, gray version of another image.
		'''
		num_diff = np.count_nonzero(cv2.absdiff(self.gray, other.gray) >= Gradient.COLOR_THRESHOLD)
		is_unique = num_diff >= self.required_differences
		return not is_unique
		
	def ToApoPalette(self, gradname):
		'''
		Converts the input colors into an ultra-fractal format
		so that apophysis and UF can use this gradient.
		'''
		
		# Gradient format stores 400 entries, despite apo only
		# using 256 entries in total...
		frac = 400.0/len(self.colors)
		
		final_grad = []
		final_grad.append(gradname + '{')
		final_grad.append('gradient:')
		final_grad.append(f'  title="{gradname}" smooth=yes')
		
		colordata = []		
		for i, color in enumerate(self.colors):
			int_color = RGBToInt(
				color.rgb[0],
				color.rgb[1],
				color.rgb[2])
			final_grad.append('  index={} color={}'.format(int(i*frac), int_color))
			
		final_grad.append('')
		final_grad.append('}')
		final_grad = '\n'.join(final_grad)
		
		# Sanity check -- string we'll save is the same as what we parse back in.
		other = Gradient.ReencodeTest(final_grad)
	
		if len(self.colors) != len(other.colors):
			msg = f'Colors in generated gradient ({len(self.colors)}) are different than reparsed gradient ({len(other.colors)}).'
			if self.colors[0] != other.colors[0]:
				msg += f' Starting colors do not match ({self.colors[0]} vs {other.colors[0]})'
			if self.colors[-1] != other.colors[-1]:
				msg += f' Ending colors do not match ({self.colors[-1]} vs {other.colors[-1]})'
			
			raise ValueError(msg)
		
		for i, (a, b) in enumerate(zip(self.colors, other.colors)):
			if a.rgb != b.rgb:
				raise ValueError(f'Mismatch sanitizing palette encode (position {i}) -- ({a.rgb} vs {b.rgb})')

		return final_grad
			
	def HasLuminanceNoise(self, lumdiff, lumcount):
		
		ALLOWED_LUM_JUMP    = lumdiff / 100.0
		TOTAL_JUMPS_ALLOWED = int(SAMPLESIZE * (lumcount/100.0))
		
		noise = 0
		run   = []
		
		for this_lab in (color.lab[0] for color in self.colors):
		
			try:
				# True is up, false is down.  We only check the history and average
				# shifte in luminance upon switching directions.
				run_direction = run[1] > run[0] 
				
				case_a =     run_direction and this_lab < run[-1]
				case_b = not run_direction and this_lab > run[-1]
				
				if case_a or case_b:
					average_delta = abs(run[-1] - run[0]) / len(run)
					if average_delta > ALLOWED_LUM_JUMP:
						noise += 1
						if noise > TOTAL_JUMPS_ALLOWED:
							return True
				del run[:-1]
				
			except:
				pass
				
			run.append(this_lab)
			
		return False
		
	
class UniqueGradientChecker(object):
	'''
	This class stores a bunch of gradients and offers functionality
	to add new gradients that are not similar to what it already
	contains.
	
	The actual criteria used for similarity is specified in the
	Gradient object itself.
	'''

	def __init__(self):
		self.unique = []
		
	def IsUnique(self, gradient):
		return not any(gradient.IsDuplicate(x) for x in self.unique)
		
	def AppendIfUnique(self, gradient):	
		'''
		Checks if the gradient passed in is unique and adds it to the
		list of unique gradients if it is.
		'''
		if self.IsUnique(gradient):
			self.unique.append(gradient)
			return True
		return False
		
	def MergeWith(self, other):
		'''
		Assuming we have a bunch of input data from another gradient checker
		that's considered unique, speed up processing by not recomparing
		these new gradients against each other as they're added in to the mix.
		'''
		
		addWhat = [x for x in other.unique if self.IsUnique(x)]
		self.unique.extend(addWhat)
