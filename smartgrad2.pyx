import os, argparse, time, multiprocessing, copy
import numpy as np
import cv2
import itertools
from queue        import Empty
from contextlib   import closing
from collections  import defaultdict, deque, Counter
from GradGui      import GuiWrapper, PreviewSelector, Displayable, SelectableObject
from CVUtils      import *
from commonutils  import *
from gradient     import *
from markov       import *

import random

MAXTHREADS = max(1, int(multiprocessing.cpu_count()-1))
THREADS    = min(MAXTHREADS, 6)
USABLE_SCREENSIZE = GetScreensize()[0]-50, GetScreensize()[1]-100

	
class EqualizationGui(GuiWrapper):
	def __init__(self, *args, **kwargs):
		GuiWrapper.__init__(self, *args, **kwargs)

	def HandleButtonClick(self, mouse, displacedMouse):
		# If we select a button, then save the selected gradient or shift the gradient list up or down a bit.
		for button in [x for x in self.buttons if x.MouseOver(mouse)]:
	
			if button.name == 'UP':
				self.MoveUp()
				return True
				
			elif button.name == 'DOWN':
				self.MoveDown()
				return True
				
			elif button.name == 'SAVE':
				return None
				
		return False
	
	
def MonitorThreadProgress(threads, queue, percentage_func, end_cond_func):

	# Data collection and monitoring ------------------------------------------
	
	remaining = len(threads)
	results   = [None for _ in threads]
	counts    = [   0 for _ in threads]
	
	# Thread management and start/finishing -----------------------------------
	
	unstarted_threads = threads[:]
	for i in range(min(len(threads), THREADS)):
		unstarted_threads.pop(0).start()
		
	# Main loop turn on.  WE GET SIGNAL. --------------------------------------
	
	print('')
	print('Progress'.center(max(14, len(threads)*7), '-'))
	while remaining > 0:
	
		t = time.time()
		try:			
			threadid, item = queue.get(True, 0.05)
			
			# Has the thread finished? ------------------------------------
			if end_cond_func(item):
				counts[threadid] = 100
			
				remaining -= 1
				results[threadid] = item
				
				# Join this thread and start any remaining threads --------
				threads[threadid].join()
				if unstarted_threads:
					unstarted_threads.pop(0).start()
					
			else:
				counts[threadid] = percentage_func(item)
					
		except Empty:
			pass
			
		collected = ' | '.join(f'{int(x):>3}%' for x in counts)
		print('\r', collected, end='')
		
	print('')
	print('Finished'.center(max(14, len(threads)*7), '-'))
	print('')
	return results
	
	
def ImageEqualizations(rgb_base, args):

	def EnhanceImage(img, all_channels, channel_perms):
		equalizations = []
		for each_perm in channel_perms:
			tmp     = img.copy()
			for elem in each_perm:
				i = all_channels.index(elem)
				tmp[:,:,i] = cv2.equalizeHist(tmp[:,:,i])

			equalizations.append((''.join(each_perm), tmp))
		return equalizations
		
	def EnhanceSaturation(img_type):
		tmp = img_type.To('hsv').img
		tmp[:,:,1] = cv2.equalizeHist(tmp[:,:,1])
		return TaggedImage(tmp, 'hsv').To(img_type.tag)
		
	def EnhanceValue(img_type):
		tmp = img_type.To('hsv').img
		tmp[:,:,2] = cv2.equalizeHist(tmp[:,:,2])
		return TaggedImage(tmp, 'hsv').To(img_type.tag)
		
	def EnhanceSatValue(img_type):
		tmp = img_type.To('hsv').img
		tmp[:,:,1] = cv2.equalizeHist(tmp[:,:,1])
		tmp[:,:,2] = cv2.equalizeHist(tmp[:,:,2])
		return TaggedImage(tmp, 'hsv').To(img_type.tag)
		
	all_enhancements = [('Original Image', rgb_base.img)]

	# HSV ---------------------------------------------------------------------
	hsv_perms = ['s', 'v', 'sv']
	for txt, img in EnhanceImage(rgb_base.To('hsv').img, 'hsv', hsv_perms):
		converted = TaggedImage(img, 'hsv').To('bgr').img
		all_enhancements.append((f'hsv_{txt}', converted))
		
	# LAB ---------------------------------------------------------------------
	raw_lab_enhancements = []
	lab_perms = ['lab', 'l', 'la', 'lb', 'ab', 'a', 'b']
	for txt, img in EnhanceImage(rgb_base.To('lab').img, 'lab', lab_perms):
		converted = TaggedImage(img, 'lab').To('bgr').img
		raw_lab_enhancements.append((f'lab_{txt}', converted))

	# Luminance-adjusted RGB --------------------------------------------------

	for txt, img in [x for x in raw_lab_enhancements]:
		all_enhancements.append((txt, img))
		if txt == 'lab_l':
			image = TaggedImage(img, 'bgr', force_rgb=True)
			sat   = EnhanceSaturation(image).img
			all_enhancements.append((f'l_{txt}_sat', sat))
	
	bgr_perms = ['rgb', 'rb', 'r', 'rg', 'g', 'gb', 'b']
	raw_rgb_enhancements = EnhanceImage(rgb_base.img, 'bgr', bgr_perms)
	
	for txt, img in raw_rgb_enhancements:	
		image = TaggedImage(img, 'bgr')
		all_enhancements.append((f'rgb_{txt}', image.img))
		
		satval = EnhanceSatValue(image).img
		all_enhancements.append((f'rgb_{txt}_satval', satval))
		
	print(list(x[0] for x in all_enhancements))

	
	# Show the equalized things so we can pick one ----------------------------	
	#displayables = [Displayable(OpenCVToPIL(cv2.medianBlur(img, 3)), name, name) for name, img in all_enhancements]
	displayables = [Displayable(OpenCVToPIL(img), name, name) for name, img in all_enhancements]
	gui = EqualizationGui(displayables, args, '')
	gui.Run()
	
	process_what = [result.name for result in gui.GetSelectedPreviews()]
	print('Selected:', process_what)
	if not process_what:
		print('No equalized image was selected.  Bailing.')	
	
	rgb_bases = [base for name, base in all_enhancements if name in process_what]
	names     = [name for name, base in all_enhancements if name in process_what]
	print(f'Number of rgb_bases: {len(rgb_bases)}: {names}')
		
	return rgb_bases[0]
		
		
def LoadImages(fname, args):
	'''
	Given a source file and some list of args, convert the image into various
	color spaces and return the results.
	'''
			
	# Ignore alpha in the input. ----------------------------------------------
	
	img  = Image.open(fname).convert('RGB')
	w, h = img.size
	changed = False
	
	# Default image resolution we'll use will be fairly small so we don't have too
	# many pixels worth of data to manage.
	
	if args.size == '-1':
		compsize = img.size[0] * img.size[1]
	else:
		compsize = [int(x) for x in args.size.split('x')]
		compsize = compsize[0] * compsize[1]
	
	# Mutual exclusion between upscaling and downscaling, since downscaling
	# is lossy and would make little sense on an image we intentionally wanted
	# to upscale.
	
	if w*h < compsize:

		# Upscaling has some type of smoothing filter that introduces more
		# of a gradient than a single color, and may be a desireable effect in upscaling
		# small images (like pixel art)
		
		if args.upscale:
			while w*h < compsize:
				w *= 1.01
				h *= 1.01
				changed = True
			w = int(w)
			h = int(h)
				
		elif w*h < 600*600:
			print('You probably should use the "--upscale" argument to see better results.')
				
	elif w*h > compsize:
	
		# Downscale the image until the area is below the desired pixel count.
		while w*h > compsize:
			w *= 0.999
			h *= 0.999
			changed = True
			
		w = int(w)
		h = int(h)
		
	if changed:
		print(f'Scaling image from "{img.size[0]}x{img.size[1]}" to "{w}x{h}"')
		img = ResizeImage(img, (w, h))
	
	# -------------------------------------------------------------------------
	
	# Before breaking up the image into colorspaces for features, we want to
	# do optional improvements on the input image to enhance parts of images
	# in various colorspaces.
	
	rgb_base = PILToOpenCV(img)
	
	if not args.base_img:
		rgb_base = ImageEqualizations(TaggedImage(rgb_base, 'bgr'), args)
									
	# Colorspace conversions --------------------------------------------------
	images = {}
	for txt, convertAs in ColorizeCVImg(rgb_base).items():
		for i, letter in enumerate(txt):
			images[f'{txt}_{letter}'] = (convertAs[:,:,i])	
			
	return images, img.size, rgb_base

	
def MasksSeemDifferent(existing_diffs, newmask, nonzero, **kwargs):
	'''
	Check if the pixels occupied by an input mask, and existing masks, seem
	different enough to be considered as separate candidates for gradient 
	color selection.
	'''

	region_percent_similar = kwargs.get('regionpercent', 0.8)
	pixel_percent_similar  = kwargs.get('pixelpercent',  0.32)
	
	min_region_thresh = 0.1 * newmask.shape[0] * newmask.shape[1]
	
	for name, (other_mask, other_nonzero) in existing_diffs.items():
	
		mincount = min(nonzero, other_nonzero)
		maxcount = max(nonzero, other_nonzero)
		
		small_pixel_count = maxcount - mincount < min_region_thresh
		
		if small_pixel_count or mincount >= maxcount * region_percent_similar:
							
			# Contains the content of both masked images at once.
			mask_space = other_mask ^ newmask
			nonzero_masked = cv2.countNonZero(mask_space)
								
			# Figure out how many pixels belong to the first and second images.
			mask_space &= newmask
			new_count = cv2.countNonZero(mask_space)
			
			if new_count < max(nonzero_masked, min_region_thresh) * pixel_percent_similar:
				return None

	return newmask, nonzero

	
def GenerateMaskedDiffImg(orig, diffImg, combined_masks, mask_color_img):
	'''
	Replace the mask white color with the maskcolor. We mask both images
	separately -- the mask itself with a color not present in the image,
	and the image itself to remove the parts we don't care about.
	'''
	
	masked_merge = cv2.bitwise_and(mask_color_img, mask_color_img, mask=combined_masks)			
	masked_img   = cv2.bitwise_or(orig,  orig,  mask=~combined_masks)
	final_merge  = cv2.bitwise_or(masked_img, masked_merge)

	return final_merge
	
	
def GenerateRGBAFromMask(src, mask):
	''' Given a source image and a mask, use the mask to create an alpha channel. '''
	rgb_to_rgba = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
	return cv2.bitwise_or(rgb_to_rgba, rgb_to_rgba, mask=mask)
		
		
class ContourGenThread(multiprocessing.Process):

	def __init__(self, thread_idx, images, orig, thresh_vals, maskcolor, queue):
		multiprocessing.Process.__init__(self)
		self.thread_idx  = thread_idx
		self.images      = images
		self.orig        = orig
		self.thresh_vals = thresh_vals
		self.maskcolor   = maskcolor
		self.queue       = queue
		
		imgtypes = list(images)
		self.filtertypes = []
		for i, name1 in enumerate(imgtypes):
			for name2 in imgtypes[i+1:]:
				self.filtertypes.append(f'{name1} {name2}')
		
	def run(self):	
		''' 
		Contourizes images and generates masks for the parts related to each contour.
		Two of the images are then merged together, and if they're unique enough compared
		to what we've already seen, are appended to the final returned dictionary.
	
		Returned is a tuple of three things:
			0) prior_diffs:  The image masks of images we want to display.
			1) merged_imgs:  The actual merged images themselves formed from the contouring
			2) named_colors: Descriptive string saying what contributed to the image.
		'''
		shape               = self.orig.shape
		mask_color_img      = np.zeros((shape[0], shape[1], 3), np.uint8)
		mask_color_img[:,:] = self.maskcolor[::-1]  # RGB --> BGR because opencv...
		
		thresh_str = '_'.join(str(x) for x in self.thresh_vals)
		contoured  = {}
		
		current_progress = 1
		ALL_PROGRESS     = len(self.images) + len(self.filtertypes)
		
		# Get contour data for all loaded images. ---------------------------------

		for name, data in self.images.items():
		
			contours = GenerateContour(
				data, 
				thresh_maxval    = self.thresh_vals[0],
				thresh_blocksize = self.thresh_vals[1],
				thresh_c         = self.thresh_vals[2])
						
			mask, img       = ExtractContourFeatures(contours, self.orig)
			contoured[name] = { 'mask':mask, 'img':img, 'name':name }
			
			self.queue.put((self.thread_idx, 100*current_progress/ALL_PROGRESS))
			current_progress += 1
				
		# Walk through the combinations of contour types that provide interesting samples of the image.
		
		prior_diffs  = {}
		merged_imgs  = {}
		
		for i, (a, b) in enumerate((x.split() for x in self.filtertypes)):
					
			# We first apply gaussian blur to fuzz up the masked region, which ends up preserving
			# certain hard edges and textures better than just applying a median filter to the data.
			# We still apply a median filter afterward because it removes very thin lines that remain
			# from the masks, which tend to be very noisy and not be very important.
			
			combined_masks = cv2.bitwise_or(contoured[a]['mask'], contoured[b]['mask'])
			
			gmblur   = cv2.GaussianBlur(combined_masks, (3,3), 0)		
			gmblur   = cv2.medianBlur(gmblur, 5)
			hardmask = cv2.multiply(gmblur, 255)
			combined_masks = hardmask
			
			# Only keep masks around if they're different enough from the existing masks we've collected.
			# We only keep existing imagery that doesn't clash with known good images.
			#
			# Keeping all masks we've seen would be greatly decrease the comparison speed since there's no
			# early abort in these operations, and as we're  already biasing the output by comparing it
			# against previous images we've seen, this would be a magically more "pure" solution either
			# for a massive amount of extra time require to process it.
			
			result = MasksSeemDifferent(prior_diffs, combined_masks, cv2.countNonZero(combined_masks))
			if result:
			
				descriptive_txt = '.'.join([thresh_str, contoured[a]['name'], contoured[b]['name']])
				prior_diffs[descriptive_txt] = result
				
				diffImg     = 255 - cv2.absdiff(contoured[a]['img'],  contoured[b]['img'])
				masked_rgb  = GenerateMaskedDiffImg(self.orig, diffImg, ~combined_masks, mask_color_img)
				masked_rgba = GenerateRGBAFromMask(masked_rgb, combined_masks)
				
				# Stored is the number of masked pixels, and the image of the mask itself.
				merged_imgs[descriptive_txt] = cv2.countNonZero(combined_masks), OpenCVToPIL(masked_rgba)
				
			self.queue.put((self.thread_idx, 100*current_progress/ALL_PROGRESS))
			current_progress += 1
		
		self.queue.put((self.thread_idx, (prior_diffs, merged_imgs)))
		
	
def interesting_adaptiveThresh():
	'''
	This attempts to be a list of interesting adaptive thresholds that maximizes the number of
	unique results we see in the pass looking for interesting regions of the image.
	
	Since each adaptive threshold / resulting set of images generated are launched on separate
	threads, the overhead of adding (or removing) these isn't as time-saving as optimizing ContourImpl.
	Removing buckets might be sufficient if you're only one or two buckets above the number of threads
	you have to process these.
	
	These numbers were empirically selected for one of two reasons:
	
		1) It produced a consistently (and subjectively) interesting set of
		   results from ContourImpl from a large number of images many times
	
		2) It seems to hit some combinations of colorspaces that aren't necessarily
		   duplicates from previous colorspaces based on running this script with
		   various images a large number of times.
	'''
	
	yield from [
		( 75, 163,  -6),
		(162,  77,  -7), 
		( 51, 163, -11),
		(110, 199, -22),
		(125,  67, -37),
		( 41,  43,  -5),
	]
	
	
def GenerateIndividualGradient(markov_cache, cache_list, args, desired):

	desired_remaining = set(desired)
	
	# Supersampling parameters --------------------------------------------
	
	samplestep = args.samplescale	
	if args.randomsample:
		samplestep = random.randint(1, args.samplescale)
		
	samples = SAMPLESIZE*samplestep
	
	# Generate the gradient -----------------------------------------------
	
	start  = random.choice(cache_list)
	colors = [start]

	# Generate enough samples ---------------------------------------------
	
	while len(colors) < samples:
	
		# Pick the next color. --------------------------------------------
		
		if args.mutation > 0 and random.random() <= args.mutation:
			next = random.choice(cache_list)
		else:
			next = markov_cache[colors[-1].Next()]
		
		colors.append(next)
		
		# If we're need certain colors to be present, then check if this color
		# is within some tolerance of the color we want.
		
		for desired_close_color in tuple(desired_remaining):
			if all(bound-args.contains_range <= target <= bound+args.contains_range for target, bound in zip(next.rgb, desired_close_color)):
				desired_remaining.discard(desired_close_color)
			
	colors = colors[::samplestep]
	return (not desired_remaining), colors, samplestep
	
	
def RandomWalk(img, desired_colors, args):

	desired_remaining = set(desired_colors)
	px = img.load()
	
	# Supersampling parameters --------------------------------------------
	
	samplestep = args.samplescale	
	if args.randomsample:
		samplestep = random.randint(1, args.samplescale)
		
	samples = SAMPLESIZE*samplestep
	
	def PickRandomLocation():
		start_x = random.randint(0, img.size[0]-1)
		start_y = random.randint(0, img.size[1]-1)
	
		if img.mode == 'RGBA':
			while px[start_x, start_y][-1] == 0:
				start_x = random.randint(0, img.size[0]-1)
				start_y = random.randint(0, img.size[1]-1)
				
		return start_x, start_y
	
	
	def NextLocation(xy):	
		x = xy[0]
		y = xy[1]
		possibilities  = [xy]
		test_positions = [
			[x-1, y-1],  # Up Left
			[x,   y-1],  # Up
			[x-1, y-1],  # Up Right
			
			[x-1, y],  # Left
			[x+1, y],  # Right
			
			[x-1, y+1],  # Down left
			[x,   y+1],  # Down
			[x-1, y+1],  # Down Right
		]
		
		for x1, y1 in test_positions:
			if (0 <= x1 < img.size[0]) and (0 <= y1 < img.size[1]):
				if not img.mode == 'RGBA' or px[x1, y1][-1] != 0:
					possibilities.append((x1, y1))
					
		return possibilities
		
	def Pixel(xy):
		rgb = px[xy]
		if img.mode=='RGBA':
			rgb = rgb[:-1]
		return MarkovNode(rgb)
		
	positions = [PickRandomLocation()]
	colors    = [Pixel(positions[-1])]
	
	# Generate enough samples ---------------------------------------------
	
	while len(colors) < samples:
	
		# Pick the next color. --------------------------------------------
		
		if args.mutation > 0 and random.random() <= args.mutation:
			positions.append(PickRandomLocation())
		else:
			positions.append(random.choice(NextLocation(positions[-1])))
			
		next = Pixel(positions[-1])
		colors.append(next)
		
		# If we're need certain colors to be present, then check if this color
		# is within some tolerance of the color we want.
		
		for desired_close_color in tuple(desired_remaining):
			if all(bound-args.contains_range <= target <= bound+args.contains_range for target, bound in zip(next.rgb, desired_close_color)):
				desired_remaining.discard(desired_close_color)
			
	colors = colors[::samplestep]
	
	return (not desired_remaining), colors, samplestep

	
def GetGradientsFrom(markov_cache, args, data, desired):

	text, img = data
	
	print('')
	print(f'Generating for "{text}"'.center(60, '-'))

	# Change the run limit a little bit if we want a specific color in the gradient,
	# as it might be harder to iterate onto. 
	
	if len(desired) == 0:   CONSEC_RUN_LIMIT = 32
	elif len(desired) == 1: CONSEC_RUN_LIMIT = 386
	else:                   CONSEC_RUN_LIMIT = 1024
	
	consec_bad_runs  = 0
	cache_list       = tuple(val for _, val in markov_cache.items())
	checker          = UniqueGradientChecker()
	
	while 1:

		# Always assume the run is bad.  It'll be reset later if it's good.
		consec_bad_runs += 1 
		
		if random.randint(0,1) == 1:
			walk = 'walk'
			valid, colors, samplestep = RandomWalk(img, desired, args)
						
		else:
			walk = 'transition'
			valid, colors, samplestep = GenerateIndividualGradient(markov_cache, cache_list, args, desired)

		if valid:
			gradient = Gradient(f'{text}_{walk}_ss{samplestep}', colors, args)
			if not args.lum or not gradient.HasLuminanceNoise(args.lumdiff, args.lumcount):
				if checker.AppendIfUnique(gradient):
					consec_bad_runs = 0
			
					'''
					colorset = set(x.rgb for x in colors)
			
					tmp = img.copy()
					px  = tmp.load()
					for y in range(tmp.size[1]):
						for x in range(tmp.size[0]):
							if px[x,y] not in colorset:
								px[x,y] = tuple(int(0.25*val) for val in px[x,y])
								
					DisplayImg(tmp)
					'''
		
		print(f'\r{len(checker.unique):>5} ({consec_bad_runs:<5})', end='')
		if consec_bad_runs >= CONSEC_RUN_LIMIT or len(checker.unique) >= args.count:
			print(' ...finished\n')
			return checker
	
	
def GenerateDifferenceImages(images, size, orig, maskcolor):
	
	'''
	Generate a list of images that have 
	'''

	prior_diffs  = {}
	merged_imgs  = {}
	
	threads = []
	queue   = multiprocessing.Queue()
	for i, thresh_vals in enumerate(interesting_adaptiveThresh()):
		threads.append(ContourGenThread(i, images, orig, thresh_vals, maskcolor, queue))

	results = MonitorThreadProgress(
		threads,
		queue,
		percentage_func = lambda x : x,
		end_cond_func   = lambda x : type(x) is tuple
	)
		
	for i, (t_prior_diffs, t_merged_imgs) in enumerate(results):

		print(f'Processing data from thread {i+1}/{len(results)}')
	
		if not prior_diffs:
			prior_diffs = t_prior_diffs
			merged_imgs = t_merged_imgs
			
		else:
			for name in t_merged_imgs:					
				if MasksSeemDifferent(prior_diffs, *t_prior_diffs[name]):
					prior_diffs[name]  = t_prior_diffs[name]
					merged_imgs[name]  = t_merged_imgs[name]
							
	return merged_imgs
				
	
def GetContourizedColors(images, size, orig, args):

	orig_pil  = OpenCVToPIL(orig).convert('RGB')
	maskcolor = GetColorNotIn(orig_pil)

	# These combinations don't work too well if images are strongly washed out, so passing through a 
	# contrast filter or light balance filter first might help stuff pop out more easily.
	
	markov_chains = {}
	imgmap        = {}
		
	# Only process things in colorspsaces if we're looking at more than the original image.
	
	process_what = [('original', orig_pil)]
	if not args.original:
	
		imgmap = GenerateDifferenceImages(images, size, orig, maskcolor)
		
		imgmap['original'] = 0, orig_pil
			
		previews     = sorted((count, name, img) for name, (count, img) in imgmap.items())
		displayables = [Displayable(img, name, f'{name} ({count})') for count, name, img in previews]
		
		gui = PreviewSelector(displayables, USABLE_SCREENSIZE, '', args)	
		gui.Run()
		process_what = [(result.name, imgmap[result.name][1]) for result in gui.GetSelectedPreviews()]

	
	queue   = multiprocessing.Queue()
	threads = []
	for i, (_, img) in enumerate(process_what):
		threads.append(MarkovThread(i, img, queue))

		
	results = MonitorThreadProgress(
		threads,
		queue,
		percentage_func = lambda x, y=orig_pil.size[1]: 100*x/y,
		end_cond_func   = lambda x : type(x) is dict
	)
	
	#results = [TestTheMarkov(orig_pil)]

	return results, process_what

	
def GenerateGradientsFromChains(chains, process_what, args, desired_colors, checker):
	'''
	Extract gradients from each of the markov objects
	'''

	# Keep track of old gradients before appending new ones
	old_unique = set(checker.unique)
	
	for i, cache in enumerate(chains):
		checker.MergeWith(GetGradientsFrom(cache, args, process_what[i], desired_colors))	
	
	new_unique = set(checker.unique)	
	
	# Remove old gradients from the new set to capture all added values
	differences = new_unique - old_unique
	if differences:
		return differences
	
	print('Quitting because no gradients were selected or no unique gradients were found.')
	exit(1)

	
