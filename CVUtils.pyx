import os, random
import cv2, pygame
import numpy as np

from math      import floor, ceil
from functools import lru_cache
from PIL       import Image, ImageFont, ImageDraw

FONT_REGISTRY = {}

class TaggedImage(object):

	@staticmethod
	def SanitizeTag(tag):
		tag = tag.upper()
		if tag == 'YCRCB':
			tag = 'YCrCb'
		return tag

	def __init__(self, img, tag, **kwargs):
		self.img = img
		self.tag = TaggedImage.SanitizeTag(tag)
		
		if tag == 'RGB':
			if not kwargs.get('force_rgb', False):
				raise ValueError('''
					It is likely you want BGR rather than RGB. OpenCV inherently
					stores images as BGR format (0xBBGGRR).  Use "force_rgb=True" if
					you know this format should be RGB and want to force it.
				'''.replace('\n', ' ').replace('\t', ''))
		
	def To(self, dst_fmt):
		dst_fmt = TaggedImage.SanitizeTag(dst_fmt)
		if dst_fmt == self.tag:
			return self
			
		full_tag = getattr(cv2, 'COLOR_{}2{}'.format(self.tag, dst_fmt))
		image    = cv2.cvtColor(self.img, full_tag)
		return TaggedImage(image, dst_fmt, force_rgb=True)
			

@lru_cache(None)
def GetScreensize():
	''' Hacky way to get screensize, lol '''
	import tkinter as tk
	root = tk.Tk()
	w = root.winfo_screenwidth()
	h = root.winfo_screenheight()
	root.destroy()
	return w, h
	
def RGBToInt(r, g, b):
	return r | (g << 8) | (b << 16)
	
def IntToRGB(val):
	return (val & 255), ((val >> 8) & 255), (val >> 16)

def DownsizeRatio(origSize, **kwargs):
	'''
	Given an original image size (w, h), and optional new width and height, return a ratio that
	preserves the original size while ensuring that the maximum dimension does not exceed newW
	or newH.
	'''
	
	w, h = origSize
	
	newW = kwargs.get('width',  None)
	newH = kwargs.get('height', None)
	
	if newW is not None and newW < w:
		ratio = newW / w
		w, h = (int(x*ratio) for x in (w, h))

	if newH is not None and newH < h:
		ratio = newH / h
		w, h = (int(x*ratio) for x in (w, h))

	return w, h


def ShapedCVImage(shape, fill=None, dtype=np.uint8):
	img = np.zeros(shape, dtype)
	if fill is not None:
		img[:,:] = fill
	return img

	
def FilledCVImage(width, height, channels, fill=None, dtype=np.uint8):
	return ShapedCVImage((height, width, channels), fill, dtype)

	
def PILToOpenCV(img):
	'''
	Handles grayscale channels, RGB, and RGBA.
	
	Reverse the array we load so OpenCV sees the colors correctly -- OpenCV assumes data is
	stored in BGR when performing colorspace conversions, while PIL loads data into an array as RGB.
	'''
	
	numpified = np.array(img)
	
	if img.mode == 'L':
		# No transformations need to be applied to individual channels.
		return numpified
		
	elif img.mode == 'RGB':
		# 0xRRGGBB (numpy conversion) --> 0xBBGGRR (opencv expectation)
		return cv2.cvtColor(numpified, cv2.COLOR_RGB2BGR)
		
	elif img.mode == 'RGBA':
		# 0xRRGGBBAA (numpy conversion) --> 0xBBGGRRAA (opencv expectation)
		return cv2.cvtColor(numpified, cv2.COLOR_RGBA2BGRA)
		
	raise ValueError(f'Unhandled image mode {img.mode}')

	
def OpenCVToPIL(img):
	'''
	Handles grayscale channels, RGB, and RGBA.
	
	Similarly to PILToOpenCV, this undoes the RGB->BGR transformation of an image before 
	converting it into a PIL image.
	'''
				
	if img.shape[-1] == 1 or len(img.shape) < 3:
		# Assume any single-depth array, or numpy array with only two dimensions, is luminance
		return Image.fromarray(img)
	
	elif img.shape[-1] == 3:
		# Assume any numpy array with three dimensions contains RGB data
		return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	
	elif img.shape[-1] == 4:
		# Assume any numpy array with four dimensions contains RGBA data
		return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
		
	raise ValueError(f'Unknown array type of {img.shape[-1]} channels will not be decoded into an image.')
		

def ResizeImage(img, ratio):
	''' Try resizing the image in opencv, and fall back to PIL if it fails. '''
	
	try:
		return cv2.resize(img, ratio, interpolation=cv2.INTER_AREA)		
	except:
		return img.resize(ratio, Image.LANCZOS)


def DisplayImg(img, **kwargs):
	''' Just a quick window display of an opencv image buffer that waits for an in-window keypress. '''
	
	title = kwargs.get('title', 'Unnamed')
	try:
		img.shape
	except:
		img = PILToOpenCV(img)
	
	if kwargs.get('channels', False):
		for i in range(img.shape[-1]):
			use_title = title + f' Channel {i}'
			cv2.namedWindow(use_title)
			cv2.moveWindow(use_title, 10,10)
			cv2.imshow(use_title, img[:,:,i])
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	else:
		cv2.namedWindow(title)
		cv2.moveWindow(title, 10,10)
		cv2.imshow(title, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
def Rescale_Contour(contour, newscale):
	''' Scale all contour components by a certain amount, so it can fit a different resolution. '''
	tmp = []
	for eachgroup in contour:
		tmp.append(np.array([eachvec*newscale for eachvec in eachgroup], dtype=np.int32))
	return tmp		


def GenerateContour(imgChannel, **kwargs):
	'''
	Given an input image channel, return a contour based on the channel provided.
	
	The input is initially passed through a bilateral filter to smooth out the image prior to being
	thresholded and converted into contours.
	
	
	Input parameters exist in two main categories:
		1) bilateralFilter   settings: [bilateral_a, bilateral_b, bilateral_c]
		2) AdaptiveThreshold settings: [thresh_maxval, thresh_blocksize, thresh_c]
		
	See the opencv documentation on these functions, or experiment on your own to see what they do.
	'''
	
	filteredChannel  = cv2.bilateralFilter(
		imgChannel,
		kwargs.get('bilateral_a', 6),
		kwargs.get('bilateral_b', 31),
		kwargs.get('bilateral_c', 61))

	
	edges = cv2.adaptiveThreshold(
		filteredChannel,
		kwargs.get('thresh_maxval', 175),
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		kwargs.get('thresh_blocksize', 155),
		kwargs.get('thresh_c', -6))
		
	# Returns (mask, contour, _)
	
	contours, hierarchy	= cv2.findContours(
		edges, 
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		
	return contours

	
	
def ExtractContourFeatures(contours, source_img):
	''' Returns both the contour mask applied to an image, as well as a source image masked by the contour shape. '''
	
	# Fill in each contour, as we don't want to erase what the contour contains.
	mask = FilledCVImage(source_img.shape[1], source_img.shape[0], 1)
	for elem in contours:
		cv2.drawContours(mask, (elem,), 0, (255,), cv2.FILLED)

	# Create an RGB image that fills only the area we want to remove with the mask color
	mask_color = FilledCVImage(mask.shape[1], mask.shape[0], 3, (255,255,255))
	
	colored_mask  = cv2.bitwise_or(mask_color, mask_color, mask=~mask)
	colored_mask &= mask_color
	
	# Isolate the parts of the source image we want to keep.
	masked_src = cv2.bitwise_or(source_img, source_img, mask=mask)
	
	# Merge the two parts together.

	masked_src |= colored_mask
	
	return mask, masked_src
		
		
def GetColorNotIn(img):
	'''
	Determine the most suitable mask color to block out unnecessary regions of an image.
	
	It attempts to use an rgb value not present in the image, followed by the least common
	color used in the image.
	'''

	allColors = { color : count for count, color in img.getcolors(img.size[0]*img.size[1]) }
	
	# Try some fairly garish colors that are easy to detect
	garish = [
		(255,   0, 157),
		(255,   1,   1),
		(  1, 255,   1),
		(  1,   1, 255),
		(  1, 255, 255),
		(255,   1, 255),
		(255, 255,   1),
	]
	for color in garish:
		if color not in allColors:
			return color
	
	# Ignore pure white and pure black
	for r in range(254, 0, -1):
		for g in range(254, 0, -1):
			for b in range(254, 0, -1):
				if (r, g, b) not in allColors:
					return (r, g, b)
					
	# Default: Return the least common color found in allColors above.
	return sorted(allColors.items(), key = lambda x: x[1])[0]


def WriteImageText(img, txt, **kwargs):
	'''
	Write white text on an image, with optional keywords:
		position -- position to write the text on the image.
		fontsize -- size of the font to render
		fill     -- Draw a black background behind the text to make it more visible.
	'''
	
	position = kwargs.get('position', (5, 5))
	fontsize = kwargs.get('fontsize', 16)
	
	BLACK = (0,0,0)
	WHITE = (255,255,255)
	
	draw = ImageDraw.Draw(img)
	
	if kwargs.get('fill', None):
		area = (
			(position[0], position[1]-1),
			(img.size[0], position[1]+(fontsize*1.25))
		)
		draw.rectangle(area, fill=BLACK)
	
	shadow_position = tuple(x+3 for x in position)
	
	global FONT_REGISTRY
	try:
		font = FONT_REGISTRY[fontsize]
	except:
		font = ImageFont.truetype("arial", fontsize)
		FONT_REGISTRY[fontsize] = font	
	
	draw.text(shadow_position, txt, font=font, fill=BLACK)
	draw.text(position,        txt, font=font, fill=WHITE)
	
	
def ButtonWithText(size, text, bgcolor):
	'''
	Creates a buttom of the expected size with some text on it. 
	It's not pretty but will get the job done.
	'''
	button = Image.new('RGB', size, bgcolor)
	size_per_letter = int(size[0]*0.6) // len(text)
	approximate_x = (size[0] // 2) - ((size_per_letter*len(text)) // 2)	
	WriteImageText(button, text, position=(approximate_x, int(size[1] * 0.2)), fontsize = size_per_letter)
	return button
	
	
def DrawInteriorBorder(img, pixels, color):
	''' Draws a border of a certain pixel width around an image, at a specified color. '''
	
	draw = ImageDraw.Draw(img)
	
	tl_s = (0,       0)
	tl_e = (pixels,  pixels)
	
	tr_s = (img.size[0]-pixels,  0)
	tr_e = (img.size[0]       ,  pixels)
	
	bl_s = (0,           img.size[1])
	bl_s = (pixels,      img.size[1])
	br   = (img.size[0], img.size[1])
	
	draw.rectangle([(0,                  0),                  (img.size[0], pixels)],      color) # Top
	draw.rectangle([(0,                  0),                  (pixels,      img.size[1])], color) # Left
	draw.rectangle([(img.size[0]-pixels, 0),                  (img.size[0], img.size[1])], color) # Right
	draw.rectangle([(0,                  img.size[1]-pixels), (img.size[0], img.size[1])], color) # Bottom

			
def GetTileModulus(imgs, imgsize, windowsize):
	'''
	Determines how many images should fit across the image window based on the input size
	and image size we expect to have.  This doesn't actually clamp it to a given width or
	height.
	'''

	minarea  = 0
	best_mod = len(imgs)
	
	# Find the result that ends up with the best width/height ratio
	for width_modulus in range(1, len(imgs)+1):
		maxw = imgsize[0] * width_modulus
		maxh = imgsize[1] * (len(imgs) // width_modulus)
		# Scale this size to window size
		
		rescaled = DownsizeRatio((maxw, maxh), width=windowsize[0], height=windowsize[1])
		area_rescaled = rescaled[0] * rescaled[1]
		
		if area_rescaled > minarea:
			minarea = area_rescaled
			best_mod = width_modulus
		
	buckets = best_mod, ceil(len(imgs) / best_mod)
	return buckets, best_mod
	
	
def TileImages(imgs, imgsize, windowsize):
	'''
	Given a bunch of images, create an image that fits a certain window size
	with these images tiled side-by-side in a fairly interesting way.
	
	All images are expected to be imgsize.
	
	NOTE:
		This currently merges images at full size and downscales, so it might
		eat up a lot of RAM if used for a large number of high-resolution images.
		This is a place to optimize in the future if this consistently becomes a problem.
	'''

	buckets, best_mod = GetTileModulus(imgs, imgsize, windowsize)
	
	pil = Image.new(imgs[0].type, (buckets[0]*imgsize[0], buckets[1]*imgsize[1]), 0x0)
	for (i, img) in enumerate(imgs):
		x = i  % best_mod
		y = i // best_mod
		fromCV = OpenCVToPIL(img[1])
		pil.paste(fromCV, (x*imgsize[0], y*imgsize[1]))
				
	downsized = DownsizeRatio(pil.size, width=windowsize[0], height=windowsize[1])
	scalar = tuple(x/y for x, y in zip(downsized, windowsize))
	
	return scalar, buckets, ResizeImage(pil, downsized)
	
	
def GetImgTiles(imgs, imgsize, windowsize):

	'''
	Given a set of input images of a certain dimension, return a bunch of smaller images
	sized such that they'd fit the window size.
	'''

	scalar, buckets, img = TileImages(imgs, imgsize, windowsize)	
		
	wscalar, hscalar = ((x*y)/z for x, y, z in zip(windowsize, scalar, buckets))
	results = []
	
	for i in range(len(imgs)):
		x = i  % buckets[0]
		y = i // buckets[0]
		
		xmin = int(x*wscalar)
		xmax = int((x+1)*wscalar)
		ymin = int(y*hscalar)
		ymax = int((y+1)*hscalar)
		
		component = img.crop((xmin, ymin, xmax, ymax))
		
		data = {
			'position' : (xmin,      ymin),
			'size'     : (xmax-xmin, ymax-ymin),
			'img'      : component,
			'name'     : imgs[i][0],
		}
		results.append(data)
	
	return results

	
def ColorizeCVImg(cv_bgr):

	imgs = {
		'bgr' : cv_bgr,
		'hsv' : cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV),
		'yrb' : cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2YCrCb),
		'lab' : cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2LAB),
	}
	
	# First component  is hue modulated around 30 degree buckets
	#  --- 1/12 * 255 is about 21 -- 30 degree hue buckets
	# Second component is inverse saturation
	# Third component  is inverse luminance
	
	imgs['3gd'] = imgs['hsv'].copy()
	imgs['3gd'][:, :, 0] =      (imgs['3gd'][:, :, 0] % 21)*12 # Hue
	imgs['3gd'][:, :, 1] = 255 - imgs['3gd'][:, :, 1] # Saturation
	imgs['3gd'][:, :, 2] = 255 - imgs['3gd'][:, :, 2] # Inverse Lum	
	

	imgs['lch'] = imgs['lab'].copy()
	lch  = np.float64(imgs['lch']) / 255.0
	aa   = np.power(lch[:,:,1], 2)
	bb   = np.power(lch[:,:,2], 2)
	imgs['lch'][:, :, 1] = np.uint8(255*np.sqrt(aa+bb))
	imgs['lch'][:, :, 2] = np.uint8(255*np.arctan2(lch[:,:,2], lch[:,:,1]))
				
	return imgs
