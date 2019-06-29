# distutils: language = c++
import time
import random
import multiprocessing

from collections  import Counter, defaultdict
from commonutils  import KeyableDefaultDict
from colorconvert import RGB_To_LAB, FromSRGB, QuantizeLAB
from colorconvert cimport RGBToInt

from CVUtils      import *
from PIL          import Image

cimport numpy as np


class MarkovNode(object):
	'''
	A bidirectional markov chain node and class.  No significant throught
	was put into optimizing this for a combination of space and speed.
	'''
	__slots__ = ('rgb', 'lab', 'transitions')
	def __init__(self, rgb):
		''' 
		Initializes the Markov Chain entry of a certain `rgb` color.
		    `rgb` 
				Must be an RGB tuple (r,g,b), and each component
				must fall in the range of [0,255].
		'''
		self.rgb  = tuple(rgb)
		rgb_float = FromSRGB(self.rgb)
		lab_float = RGB_To_LAB(rgb_float)
		self.lab  = QuantizeLAB(lab_float)
		
		self.transitions = []

	def ConnectTo(self, other):
		self.transitions.append(other.rgb)
		other.transitions.append(self.rgb)
		
	def Next(self):
		''' Returns an object that this node connects to based on their weights. '''
		return random.choice(self.transitions)
		
		
def GenerateMarkovCacheFromImage(img, markov_cache):
	'''
	Given a source image, this returns a markov chain state that can be used in other functions
	to generate random gradients.
	
	Assume fixed radius of 1. There's two major motivations for this: 
	
		- General Aesthetic Niceness: 
			Having a larger pixel radius typically makes gradients a lot busier, which 
			negatively affects small source images. Instead of radius being a parameter,
			consider using the `samplescale` parameter to skip over transitions that often
			occur in these gradients.
			
		- Speed:
			The larger the radius, the more pixels an individual color needs to associate
			itself with the colors around it. A radius of 1 can be optimized to only require
			four total checks (down/left, right, up, and up/right), whereas naive radius checks
			require [8 (3x3 - 1)], [24 (5x5 - 1)], [48 (7x7 - 1)] neighbors. Even if optimized
			versions cut down checks in half, this adversely affects larger images because more
			pixels have to be checked in a larger radius, which is a worst-case when it comes
			to speed.
			
	Simplicity also is a factor, because it's less code too. The 1 radius fanout can be
	optimized to four checks, under the assumption that pixels we've already processed
	pixels in the row above and prior to this pixel
	
	 | . | . | . |
	 | . | x | o |
	 | o | o | o |
	
	Edge cases aren't handled very frequently, and wrapping this in a try/except block 
	seems to be fairly performant, all things considered.
	
	'''
		
	size = img.size
	px   = img.load()
	
	# Doing this ensures no pixel values need to be created when
	# we iterate over the image.
	for y in range(size[1]):
		for x in range(size[0]):
			markov_cache[px[x, y]]
	
	# Optimized if we're using RGB, since we don't need to check alpha values
	if img.mode == 'RGB':
		
		next_row = [markov_cache[px[x,0]] for x in range(size[0])]
		
		for y in range(size[1]):
			yield y
			current_row = next_row
		
			try:    next_row = [markov_cache[px[x,y+1]] for x in range(size[0])]
			except: next_row = None
			
			for x in range(size[0]):
				current_px = current_row[x]
				current_px.ConnectTo(current_px)
			
				try:    current_px.ConnectTo(current_row[x+1])
				except: pass
			
				bin = (x-1, x, x+1) if x else (x, x+1)
				for x1 in bin:
					try:    current_px.ConnectTo(next_row[x1])
					except: pass

				
	# We have to branch on alpha values in RGBA mode
	elif img.mode == 'RGBA':	
		for y in range(size[1]):
			yield y
			for x in range(size[0]):
				r1, g1, b1, a1 = px[x,y]
				if a1 != 0:
					color1 = markov_cache[r1,g1,b1]
					for xy in ((x,y), (x+1,y), (x-1,y+1), (x,y+1), (x+1, y+1)):
						try:
							r2, g2, b2, a2 = px[xy]
							if a2 != 0:
								color1.ConnectTo(markov_cache[r2,g2,b2])
						except IndexError:
							continue
							
		# Prune unused values (those with no pixels that weren't totally opaque
		for k, v in list(markov_cache.items()):
			if not v.transitions:
				del markov_cache[k]



def MarkovTestFunction(img, cache):

	size = img.size
	px   = img.load()

	for y in range(size[1]):
		for x in range(size[0]):
			# Doing this ensures no pixel values need to be
			# created when we iterate over the image.
			cache[px[x, y]]
	
	next_row = [cache[px[x,0]] for x in range(size[0])]
	
	for y in range(size[1]):
		current_row = next_row
	
		try:    next_row = [cache[px[x,y+1]] for x in range(size[0])]
		except: next_row = None
		
		for x in range(size[0]):
			current_px = current_row[x]

			current_px.ConnectTo(current_px)
		
			try:    current_px.ConnectTo(current_row[x+1])
			except: pass
		
			bin = (x-1, x, x+1) if x else (x, x+1)
			for x1 in bin:
				try:    current_px.ConnectTo(next_row[x1])
				except: pass

				
def TestTheMarkov(img):

	'''
	img = Image.new('RGB', (3, 3), 0x0)
	px = img.load()
	for y in range(img.size[1]):
		print('|', end=' ')
		for x in range(img.size[0]):
			val = y*img.size[0]+x
			px[x,y] = val,val,val
			print(val, '|', end=' ')
		print('')
	'''
	
		
	cache1 = KeyableDefaultDict(MarkovNode)
	t = time.time()
	MarkovTestFunction(img, cache1)
	print(f'{time.time() - t:.3f}s')
	exit(0)
	
	cache2 = KeyableDefaultDict(MarkovNode)
	list(GenerateMarkovCacheFromImage(img, cache2))
	
	CompareMarkovCaches(cache1, cache2)
	exit(0)
	return cache1
				
							
def CompareMarkovCaches(a, b):
	
	a_keys = set(a)
	b_keys = set(b)
	if a_keys != b_keys:
		print('Keys are not similar between both caches!')
		print('\tNew:', a_keys)
		print('\tOld:', b_keys)
		exit(1)
		
	for key, obj in a.items():
		connections1  = Counter(obj.transitions)
		connections2  = Counter(b[key].transitions)
		
		if connections1 != connections2:
			a1 = { k[0] : v for k, v in connections1.items() }
			b1 = { k[0] : v for k, v in connections2.items() }
		
			print('\t',   key[0])
			print('\t\tNew:', a1)
			print('\t\tOld:', b1)
				
	
class MarkovThread(multiprocessing.Process):

	def __init__(self, index, img, queue):
		multiprocessing.Process.__init__(self)
		self.idx    = index
		self.img    = img
		self.queue  = queue

	def run(self):
	
		markov_cache = KeyableDefaultDict(MarkovNode)
		for y in GenerateMarkovCacheFromImage(self.img, markov_cache):
			self.queue.put((self.idx, y))
					
		self.queue.put((self.idx, dict(markov_cache)))	
	