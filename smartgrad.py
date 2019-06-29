import os
import argparse
from smartgrad2 import *
from gradient   import *
from GradGui    import GuiWrapper, PreviewSelector, Displayable, SelectableObject


def MainLoopTurnOn(args, desired_colors):

	images, size, cv_orig = LoadImages(args.input, args)
	
	if args.export: savedir = None
	else:           savedir = args.out
	
	# Cache the checker and chains to reuse conveniently
	chains, process_what  = GetContourizedColors(images, size, cv_orig, args)
	checker = UniqueGradientChecker()
	
	unchanging_args = {'input', 'out', 'size', 'upscale', 'original', 'base_img'}
	
	while 1:

		generated = GenerateGradientsFromChains(chains, process_what, args, desired_colors, checker)			
		grads = sorted(generated, key=lambda x : x.for_sort)

		# Some silly defaults to make things more interesting
		if args.samplescale == 1:
			args.randomsample = True
			args.samplescale  = 2
	
		displayables = [Displayable(x.GetImage(), fname=x.text, text=x.text, grad=x) for x in grads]	
		display      = GuiWrapper(displayables, args, savedir)
	
		display.Run()

		
		print('Do what?  [ctrl+c to quit, enter to regenerate, "change" to alter arg values]')
		result = input('  => ').lower().strip()
		if result.startswith('c'):
			while 1:
				print('')
				maxkey = max(len(k) for k in args.__dict__)
				for k, v in sorted(args.__dict__.items()):
					if k not in unchanging_args:
						print(f'{k:<{maxkey}} : {v} ({type(v)})')
				print('[ctrl+c to quit, enter to regenerate more]')
				result = input('  [enter option, followed by value]  => ').strip().lower()
				if result:
					try:
						key, value = [x.strip() for x in result.lower().split()]
						keyval     = getattr(args, key)
						if type(keyval) is bool:
							if   value[0] == 'f': keyval = False
							elif value[0] == 't': keyval = True
							setattr(args, key, keyval)
						else:
							setattr(args, key, keytype(value))
							
					except Exception as e:
						print('Error encountered, please try again.')
						print(e)
				else:
					break
		
		elif result.startswith('q'):
			break


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	# Input filepath arguments ------------------------------------------------
	
	parser.add_argument('-i', '--input', type=str, required=True, nargs='+',
		help='File to work gradient magic on.')
		
	output_type = parser.add_mutually_exclusive_group(required=True)
		
	output_type.add_argument('-o', '--out',   type=str, nargs='+',
		help='Output directory for files that have been turned into gradients')				
		
	output_type.add_argument('-e', '--export', action='store_true',
		help='Export a gradient format instead of saving as a file.')
	
	# Image parsing / processing options --------------------------------------
	
	parser.add_argument('-s', '--size', type=str, default='1200x1200', help='''
		Format: `<width>x<height>`. "-1" uses source image size. 
		Since contour extraction does occur differently on smaller images than larger images,
		smaller resolutions will capture more features of the image while larger resolutions
		focus explicitly on the feature overlap and might not be as useful.
		Setting this value to "-1" makes it the original image size.''')
		
	parser.add_argument('-u',  '--upscale',  action='store_true', help='''
		Upscale the image area to match the `size` parameter, which helps
		very small images have a wider color space and less banding.''')
	
	parser.add_argument('-orig', '--original', action='store_true',
		help='Only use the original image, do not bother trying to break up components.')
		
	parser.add_argument('-b',    '--base_img', action='store_true',
		help='Skip image processing step at the start and only use the source image.')

	# Markov generation options -----------------------------------------------
	
	parser.add_argument('--contains', action='append', help='''
		Require a color similar to this hex value (0xRRGGBB) or comma-separatred integers (r,g,b) 
		to be present in generated gradients. If multiple values are specified, gradients will
		need to contain all mentioned elements.''')
		
	parser.add_argument('--contains_range', type=int, default=32,
		help='Delta allowed between a "contains" color and gradient color to be considered good enough.')
	
	parser.add_argument('-m', '--mutation', type=float, default=0.003333, help='''
		Randomly choose a new gradient value if a random roll is above this.
		This can help introduce more colors into gradients that can be isolated, 
		but also can introduce a lot more noise.''')
	
	parser.add_argument('-c', '--count', type=int, default=256,
		help='Upper bound of gradients generated from markov chains.')
	
	# Supersampling options during gradient generation ------------------------
	
	parser.add_argument('-ss', '--samplescale', type=int, default=0, help='''
		This helps larger images, blurred images, or images with large transitional regions
		have more colors and bands.  It basically generates a gradient that's <n> times longer,
		but only picks every <n>th element.''')
		
	parser.add_argument('-rs', '--randomsample', action='store_true', help='''
		By default, supersampling occurs at the `--samplescale` level for each generated gradient.
		This randomizes it between 1 and `--samplescale`.''')
		
	# Similarity filtering options --------------------------------------------
		
	parser.add_argument('--gt', type=int, default=20,
		help='Percentage of mismatches between gradients required before images are considered different.')
		
	parser.add_argument('--gw', type=int, default=9,
		help='Each median entry must be different than its location and this many surrounding locations from other medians to be considered unique.')
		
	parser.add_argument('-ld', '--lumdiff', type=int, default=15,
		help='If two gradient luminance values differ by this many percent, it is considered "noisy"')
		
	parser.add_argument('-lc', '--lumcount', type=int, default=6,
		help='If we see "noisy" luminance bands at least this percentage of times, throw out the gradient.')
		
	parser.add_argument('-l', '--lum', action='store_true',
		help='Perform lumiance jump filtering on the input using `--lumcount` and `--lumdiff` values to define the luminance noise parameters.')
		
	# Parse me! ---------------------------------------------------------------

	args = parser.parse_args()
	
	# Sanitize me! ------------------------------------------------------------
	
	args.input = ' '.join(args.input)
	
	if args.samplescale == 0:
		args.samplescale  = 2
		args.randomsample = True
	
	if not os.path.exists(args.input):
		raise ValueError(f'Invalid filepath:', args.input)

	if args.out:
		args.out = ' '.join(args.out)
		if not os.path.exists(args.out):
			raise ValueError(f'Invalid filepath:', args.out)
			
	desired_colors = []
	if args.contains:
		for elem in (x.lower() for x in args.contains):
			
			# Hex string ----------------------------------------------------------
			if elem.startswith('0x'):
				txt = elem[2:]
				r, g, b = [int(txt[x*2 : x*2 + 2], 16) for x in range(3)]
			
			# Assumed comma-separated values --------------------------------------
			else:
				r, g, b = [int(x) for x in elem.split(',')]
			
			desired_colors.append((r,g,b))
			print(f'Gradient must contain RGB color similar to: ({r:>3}, {g:>3}, {b:>3})')
		
	MainLoopTurnOn(args, desired_colors)