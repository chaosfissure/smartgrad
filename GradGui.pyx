import os, colorsys, time
import cv2

from commonutils import *
from collections import namedtuple
from math        import ceil
from PIL         import Image, ImageDraw, ImageEnhance
from CVUtils     import *

SelectableInputs = namedtuple('SelectableInputs', 'inputs selected unselected')


def clipboard_set(text):
	temporary_file = '__used__by__grad__gui__export__'
	with open(temporary_file, 'w') as f:
		f.write(text)
	os.system(f'clip < {temporary_file}')
	
	while os.path.exists(temporary_file):
		time.sleep(0.33)
		try:
			os.remove(temporary_file)
			print('File removed!')
			return
		except Exception as e:
			print(f'Could not remove {temporary_file} ({e}), trying again...')
		
		
class FontCache(object):
	'''
	Writing things using fonts can take a surprisingly long amount of time.
	This aims to cache individual letters or sequences of letters so that
	we can alpha blend images rather than writing fonts on things over and
	over again.
	'''

	def __init__(self, fontsize, **kwargs):
		from math import log
		
		self._spacing = 3
		self._letters = {}
		self._colors  = {}
		self._img     = Image.new('L', (fontsize*4, fontsize*4), (0,))
		self._font    = ImageFont.truetype(kwargs.get('font', 'arial'), fontsize)
		
	def _GetLetter(self, letter):
		if letter not in self._letters:
			img = self._img.copy()
			ImageDraw.Draw(img).text((0,0), letter, font=self._font, fill=(255,))
			
			# We want to crop off extraneous width-wise spacing. Even if the images
			# themselves are all the same size (for pasting consistency), we want
			# letters to be aligned in a reasonably nice way.
			
			seen_horiz = self._spacing * 2
			px = img.load()

			if letter != ' ':		
				for x in range(img.size[0]):
					for y in range(img.size[1]):
						if px[x,y]:
							seen_horiz = x + self._spacing
							break

			# Clone the cropped version of the image. -----------------------------
			self._letters[letter] = (img, seen_horiz)
		return self._letters[letter]		
		
	def WriteOn(self, image, position, text, color):

		try:
			lettercolor = self._colors[color]
		except:
			lettercolor = self._colors.setdefault(color, Image.new('RGB', self._img.size, color))
		
		x_offset = position[0]
		for char in text:
			letter, offset = self._GetLetter(char)
			image.paste(lettercolor, (x_offset, position[1]), mask=letter)
			x_offset += offset
					
					
		
class Drawable(object):

	def __init__(self, position, size):
		self.position = position
		self.size     = size
		
	# -----------------------------------------------------------------------------------------------	
	# Check if we're inside the world or not
	# -----------------------------------------------------------------------------------------------
	def InsideWorld(self, worldPosition, screenSize):
		xmin = worldPosition[0]
		ymin = worldPosition[1]
		xmax = worldPosition[0] + screenSize[0]
		ymax = worldPosition[1] + screenSize[1]
		
		if self.position[0] > xmax: return False
		if self.position[1] > ymax: return False
		if self.position[0] + self.size[0] < xmin: return False
		if self.position[1] + self.size[1] < ymin: return False
		return True
		
	# -----------------------------------------------------------------------------------------------	
	# Draw to screen
	# -----------------------------------------------------------------------------------------------
	def Blit(self, screen, worldOffset, screenSize):
		print('Blit not implemented for', self)
		
	# -----------------------------------------------------------------------------------------------	
	# Is the mouse hovering over this object?
	# -----------------------------------------------------------------------------------------------
	def MouseOver(self, position):
	
		mx, my = position
	
		# Are we even inside the rectangle?
		xmin, ymin = self.position
		xmax = xmin + self.size[0]
		ymax = ymin + self.size[1]
		
		if mx < xmin or my < ymin or mx > xmax or my > ymax:
			return False
		return True

		
class SelectableObject(Drawable):

	SELECTED   = 'default'
	UNSELECTED = 'unhover'

	# -----------------------------------------------------------------------------------------------
	# Load a PIL image into a Pygame surface that's easier to cache.
	# -----------------------------------------------------------------------------------------------
	@staticmethod
	def PygameLoad(image, max_area):
	
		ratio = DownsizeRatio(image.size, width=max_area[0], height=max_area[1])
		
		if ratio != image.size:
			image = image.resize(ratio, Image.LANCZOS)

		return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

	# -----------------------------------------------------------------------------------------------
	# All the inits plz
	# -----------------------------------------------------------------------------------------------
	def __init__(self, name, position, size, selected_pil, unselected_pil, wrapped=None):
	
		Drawable.__init__(self, position, size)
	
		self.name            = name
		self.images          = {}
		self.state           = SelectableObject.UNSELECTED
		self.cached_surfaces = {}
		self._active_image   = None
		self.wrapped         = wrapped # Hold, say, a gradient object
			
		self.RegisterState(selected_pil,   SelectableObject.SELECTED)
		self.RegisterState(unselected_pil, SelectableObject.UNSELECTED)
		self.SetState(SelectableObject.UNSELECTED)
		
	# -----------------------------------------------------------------------------------------------
	# Regiseter an image associated with a certain state
	# States are not really well-thought out, but this code wasn't designed for scalable use
	# -----------------------------------------------------------------------------------------------
	def RegisterState(self, image, statename):

		img = SelectableObject.PygameLoad(image, self.size)		

		if image.mode == 'RGBA':
			surface = pygame.Surface(img.get_size(), flags=pygame.SRCALPHA)
			surface.blit(img, (0, 0))
			surface = surface.convert_alpha()

		else:
			surface = pygame.Surface(img.get_size())
			surface.blit(img, (0, 0))
				
		
		self.images[statename]          = img
		self.cached_surfaces[statename] = surface
			
	# -----------------------------------------------------------------------------------------------
	# Set the state of the image, purging any existing states.
	# -----------------------------------------------------------------------------------------------
	def SetState(self, state=None):
	
		if state is None:
			if self.state == SelectableObject.UNSELECTED: 
				self.SetState(SelectableObject.SELECTED)
			else:              
				self.SetState(SelectableObject.UNSELECTED)
		
		else:
			self.state         = state
			self._active_image = self.images[self.state]
			
	# -----------------------------------------------------------------------------------------------
	# Draw the image on a screen
	# -----------------------------------------------------------------------------------------------
	def Blit(self, screen, worldPosition, screenSize):
	
		if self.InsideWorld(worldPosition, screenSize):
		
			rect = self.cached_surfaces[self.state].get_rect()
			rect.topleft = tuple(x-y for x, y in zip(self.position, worldPosition))
			
			screen.blit(self.cached_surfaces[self.state], rect)

			
class HoverableButton(SelectableObject):
	def __init__(self, name, img, position, size):
		alphablend = img.convert('RGBA')
		alphablend.putalpha(128)
		SelectableObject.__init__(self, name, position, size, img, alphablend)
			
			
class Displayable(object):
	def __init__(self, src_img, fname, text, **kwargs):
		self.img   = src_img
		self.fname = fname
		self.text  = text
		self.extra = kwargs

		
class GuiWrapper(object):

	HEIGHT_OFFSET = 8
	WIDTH_OFFSET  = 8

	MOUSE_POLYS = (
		(0, 0),
		(0, 8),
		(1, 7),
		(4, 10),
		(5, 10),
		(5, 9),
		(4, 8),
		(4, 6),
		(6, 6),
		(6, 5),
		(1, 0),
	)	
		
	def Draw(self):
		for button in self.buttons:
			button.Blit(self.screen, (0, 0), self.size)

		height_offset = self.HeightOffset()
		for i, grad in enumerate(self.grads):
		
			# Don't render stuff that's offscreen for speed purposes.
			if abs(height_offset - grad.position[1]) > self.size[1]:
				continue
				
			grad.Blit(self.screen, (0, height_offset), self.size)
			
		# Draw a fake thing representing a mouse so we know where the cursor is on 
		# large displays, since SDL doesn't play nicely with high DPS and zoomed-up
		# icon/text/window sizes on windows.
		if self.size[0]*self.size[1] > 1920*1080:
			mouse, _ = self.GetMousePositions()
			mouse = np.array(mouse)
			
			# [(x, y), (w, h)]
			MOUSE_SCALAR = 2
			polys = [
				tuple(int(vert*MOUSE_SCALAR+pos) for vert, pos in zip(poly, mouse)) for poly in GuiWrapper.MOUSE_POLYS
			]
			
			pygame.draw.polygon(self.screen, 0xFFFFFF, polys)
			pygame.draw.polygon(self.screen, 0x000000, polys, 2)
			
	def MoveUp(self, amount=1):
		self.offset = max(0, self.offset-amount)
	
	def MoveDown(self, amount=1):
		self.offset = min(self.maxOffset, self.offset + amount)
		
	def UnsetGradients(self):
		for grad in self.grads:
			grad.SetState(SelectableObject.UNSELECTED)
	
	def GetSelectedGrad(self):
		for grad in self.grads:
			if grad.state == SelectableObject.SELECTED:
				return grad
		return None
		
	def HeightOffset(self):
		return self.offset*(self.gradHeight + GuiWrapper.HEIGHT_OFFSET)	
	
	def GetMousePositions(self):
		mouse     = pygame.mouse.get_pos()
		displaced = (mouse[0], mouse[1] + self.HeightOffset())
		return mouse, displaced
		
	def HandleKeyDown(self, event):
	
		if   event.key == pygame.K_DOWN:     self.MoveDown()
		elif event.key == pygame.K_UP:       self.MoveUp()
		elif event.key == pygame.K_PAGEDOWN: self.MoveDown(self.page_jump)
		elif event.key == pygame.K_PAGEUP:   self.MoveUp(self.page_jump)
		elif event.key == pygame.K_HOME:     self.offset = 0
		elif event.key == pygame.K_END:      self.offset = self.maxOffset
		
		elif event.key == pygame.K_ESCAPE:
			return False
			
		return True

		
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
			
				grad = self.GetSelectedGrad()
				if not grad:
					print('No gradient is selected.  Ignoring.')
					return True
				
				if not self.dir:
					# Just plug the thing on the clipboard instead.
					palette = grad.wrapped.extra['grad'].ToApoPalette(grad.name)
					clipboard_set(palette)
					return True
				
				else:
						
					fname = input('Save as what? (enter "ignore" to cancel) => ').strip()
					if fname.lower() == 'ignore':
						print('Ignoring saving of selected gradient...')
						return True
					
					else:
						ugrdir = os.path.join(self.dir, 'ugr')
						if not os.path.exists(ugrdir):
							os.makedirs(ugrdir)
					
						img   = os.path.join(self.dir, f'{fname}.png')
						ugr   = os.path.join(ugrdir,   f'{fname}.ugr')
					
						if os.path.exists(img) or os.path.exists(ugr):
							print('Please choose another name or delete the existing file with the same name...')
							return True
						
						else:
							palette = grad.wrapped.extra['grad'].ToApoPalette(fname)
							with open(ugr, 'w') as f:
								f.write(palette)
								print('Saved!')
								return True
					
		return False
		
	def HandleGradientClick(self, mouse, displacedMouse):
		for grad in self.grads:
			if grad.MouseOver(displacedMouse):
				self.UnsetGradients()
				grad.SetState(SelectableObject.SELECTED)
				return True
		return False
				
	def HandleMouseDown(self, event):
		mouse, displacedMouse = self.GetMousePositions()
		
		# Check if any buttons need to be applied.
		handle_button = self.HandleButtonClick(mouse, displacedMouse)
		if handle_button is None: return None
		if handle_button:         return True
		
		# Check if any of the gradients should have mouse click effects applied to them.
		if self.HandleGradientClick(mouse, displacedMouse):
			return True
			
		return False
		
	def HighlightButtons(self):
	
		mouse, displacedMouse = self.GetMousePositions()
		
		for button in self.buttons:		
			mode = SelectableObject.UNSELECTED
			if button.MouseOver(mouse):
			
				if button.name == 'DOWN':
					if self.offset < self.maxOffset:
						mode = SelectableObject.SELECTED
						
				elif button.name == 'UP':
					if self.offset > 0:
						mode = SelectableObject.SELECTED
				
				else:
					mode = SelectableObject.SELECTED
							
			button.SetState(mode)		

	def HandleEvents(self):
	
		mouse = pygame.mouse.get_pos()
		for event in pygame.event.get():
		
			if event.type == pygame.QUIT:
				return False
		
			# Check and see if the mouse clicks one of our buttons.
			elif event.type == pygame.MOUSEBUTTONDOWN:
				handle_mouse = self.HandleMouseDown(event)
				if handle_mouse is None:
					return False
				break
				
			elif event.type == pygame.KEYDOWN:
				if not self.HandleKeyDown(event):
					return False
					
		# Highlight buttons our mouse is over
		self.HighlightButtons()
		return True
	
	def GetSelectedPreviews(self):
		return [x for x in self.grads if x.state == SelectableObject.SELECTED]
	
	def __init__(self, imgs, args, dir, **kwargs):
	
		self.args = args
		minwidth = kwargs.get('minwidth', 3)
		maxwidth = kwargs.get('maxwidth', 5)
		self.dir = dir
		self.page_jump = None
		
		saveImg = ButtonWithText((600, 200), 'SAVE', (32, 64, 255))
		
		arrowImg = Image.new('RGB', (300, 100), (32, 64, 255))
		draw = ImageDraw.Draw(arrowImg)
		points = [
			(100, 10),
			(200, 10),
			(200, 50),
			(250, 50),
			(150, 90),
			( 50, 50),
			(100, 50),
			(100, 10),
		]
		
		draw.polygon(points, fill=(128, 192, 255), outline=(64, 128, 255))
	
		os.environ['SDL_VIDEO_WINDOW_POS']     = '0, 30'
		os.environ['SDL_WINDOW_ALLOW_HIGHDPI'] = '1' # Doesn't seem to do anything, sadly.
		
		self.size       = DownsizeRatio(GetScreensize(), height=min(GetScreensize()[1]-100, 1080))
		self.offset     = 0
		self.maxOffset  = 0
		self.gradHeight = 0
		self.buttons    = []
		self.grads      = []

		self.IMG_W_SCALAR     = 0.85
		self.USABLE_SCREEN_W  = int(self.size[0]*self.IMG_W_SCALAR)

		# Size of the original image ------------------------------------------
		gradsize = imgs[0].img.size
		
		# Figure out how many wide we can support in the window. --------------
		across = self.USABLE_SCREEN_W // (gradsize[0] + GuiWrapper.WIDTH_OFFSET)
		across = max(minwidth, min(maxwidth, across))
		
		# Re-adjust the image sizze -------------------------------------------
		gradsize = DownsizeRatio(
			gradsize, 
			width  = (self.USABLE_SCREEN_W // across) - GuiWrapper.WIDTH_OFFSET,
			height = self.size[1]+GuiWrapper.HEIGHT_OFFSET)
				
		self.gradWidth, self.gradHeight = gradsize
		
		# Initialize pygame ---------------------------------------------------
		
		print('Initializing pygame...', end='')
		pygame.init()
		self.screen = pygame.display.set_mode(self.size, pygame.DOUBLEBUF)
		pygame.display.set_caption('Gradient Generator')
		print('initialized.')
		
		# Load gradients into pygame surfaces ---------------------------------
		
		font_cache     = FontCache(kwargs.get('fontsize', 20))
		blank_for_text = Image.new('RGB', (self.gradWidth, 36), 0x0)
		
		for i, grad in enumerate(imgs):
		
			print(f'\rConverting previews to textures: {i+1:>5} / {len(imgs)}', end='')
		
			x = i  % across
			y = i // across
			
			adjust_x =    x  * (self.gradWidth  + GuiWrapper.WIDTH_OFFSET)
			adjust_y =    y  * (self.gradHeight + GuiWrapper.HEIGHT_OFFSET) + GuiWrapper.HEIGHT_OFFSET
			next_y   = (y+1) * (self.gradHeight + GuiWrapper.HEIGHT_OFFSET)
			
			if adjust_x == 0 and next_y > self.size[1]:
				self.maxOffset += 1
				
				# This should only occur on the first thing that spills out of the screen.
				if self.page_jump is None:
					self.page_jump = max(1, y-1)
				
			adjust_x += GuiWrapper.WIDTH_OFFSET // 2
			grad.img  = grad.img.resize(gradsize)
				
			# Assume all images are the same height
			position = adjust_x, adjust_y
			max_area = self.gradWidth-10, self.gradHeight
			
			display_str = f' ↓ {grad.text} ↓ ({i+1}/{len(imgs)})'
			
			# No idea why cython can't handle this using tuples directly
			WHITE = tuple([255,255,255] if grad.img.mode == 'RGB' else [255, 255, 255, 255])
			RED   = tuple([255,  0,  0] if grad.img.mode == 'RGB' else [255,   0,   0, 255])
			BLACK = tuple([  0,  0,  0] if grad.img.mode == 'RGB' else [  0,   0,   0, 255])
			
			

			# Unselected state ------------------------------------------------
			unselected = grad.img.copy()
			unselected.paste(blank_for_text)
			DrawInteriorBorder(unselected, 4, WHITE)
			
			font_cache.WriteOn(unselected, (6,6), display_str, (255,255,255))
			
			# Selected state --------------------------------------------------
			selected = unselected.copy()
			DrawInteriorBorder(selected,  4, RED)
			
			selectable = SelectableObject(grad.fname, position, max_area, selected, unselected, grad)
			selectable.SetState(SelectableObject.UNSELECTED)
			self.grads.append(selectable)

		print(' ...finished.')	

		# Load the buttons on the right side of the screen --------------------
	
		btnXMin  = self.USABLE_SCREEN_W
		btnWidth = self.size[0] - btnXMin
		_, saveHeight  = DownsizeRatio(saveImg.size,  width=btnWidth)
		_, arrowHeight = DownsizeRatio(arrowImg.size, width=btnWidth)
		save_ypos  = self.size[1] // 2
		save_ypos -= saveHeight   // 2
		down_ypos  = self.size[1] - arrowHeight
		
		saveImg   = ResizeImage(saveImg,  (btnWidth, saveHeight))
		downarrow = ResizeImage(arrowImg, (btnWidth, arrowHeight))
		uparrow   = downarrow.rotate(180)
		
		save = HoverableButton('SAVE', saveImg,   (btnXMin, save_ypos), (btnWidth, saveHeight))
		down = HoverableButton('DOWN', downarrow, (btnXMin, down_ypos), (btnWidth, arrowHeight))
		up   = HoverableButton('UP',   uparrow,   (btnXMin, 0),         (btnWidth, arrowHeight))
		# Save button (makes an assumption that the window is tall enough)

		self.buttons += [save, down, up]		
			
	def Run(self):
	
		for event in pygame.event.get():
			pass
		
		keepalive = True
		while keepalive:
		
			# FPS limiter to prevent gpu luls
			pygame.time.Clock().tick(30)
			
			# Handle events
			keepalive = self.HandleEvents()
			
			# Redraw the screen
			self.screen.fill((0,0,0))
			self.Draw()
			pygame.display.update()

		pygame.quit()

		
class PreviewSelector(GuiWrapper):

	def Draw(self):
		# Fill in the image section of the screen with the desired bg color
		fill_w = self.USABLE_SCREEN_W
		pygame.draw.rect(self.screen, self.bgcolor, pygame.Rect(0, 0, fill_w, self.size[1]))
		
		for button in self.colorbtns:
			button.Blit(self.screen, (0, 0), self.size)
		
		GuiWrapper.Draw(self)
		
	def SelectBGColor(self, picked_btn):
		for btn in self.colorbtns:
			if btn != picked_btn:
				btn.SetState(SelectableObject.UNSELECTED)
			else:
				btn.SetState(SelectableObject.SELECTED)
				self.bgcolor = tuple(map(int, btn.name.split(',')))
				
	def __init__(self, imgs, size, txt, args, **kwargs):
	
		GuiWrapper.__init__(self, imgs, args, '', **kwargs)
	
		generateBtn     = ButtonWithText((600, 200), 'Use Selected', (32, 64, 255))
		tmp_btns        = []
		self.contourBtn = None
		self.bgcolor    = (160, 160, 160)
		self.colorbtns  = []
		
		color_xmin = 0
		color_ymin = 0
		color_ymax = 0
		btn_width  = 0
	
		for btn in self.buttons:
		
			# Replace the save button with a 'next contour' button
			if btn.name == 'SAVE':
				generateBtn     = ResizeImage(generateBtn, btn.size)
				self.contourBtn = HoverableButton('NEXT_CONTOUR', generateBtn, btn.position, btn.size)
				tmp_btns.append(self.contourBtn)				
			else:
				tmp_btns.append(btn)
				
			# Use the up button as a key to make a background selection buttons
			if btn.name == 'UP':
				color_xmin = btn.position[0]
				color_ymin = btn.position[1] + btn.size[1]
				btn_width  = btn.size[0]
			if btn.name == 'SAVE':
				color_ymax = btn.position[1]
			
		# Generate background color changing buttons.
			
		ranges = (0, 64, 128, 192, 255)
		colors = [(r, g, b) for r in ranges for g in ranges for b in ranges]
		colors.sort(key=lambda x:colorsys.rgb_to_hsv(*x))
		
		BTNS_WIDE = 4
		bg_btn_w  = btn_width//BTNS_WIDE
		
		btns_tall         = int(ceil(len(colors) / BTNS_WIDE))
		height_pxls       = color_ymax - color_ymin
		height_per_button = height_pxls // btns_tall

		for i, color in enumerate(colors):
		
			xpos     = color_xmin + (bg_btn_w          * (i  % BTNS_WIDE))
			ypos     = color_ymin + (height_per_button * (i // BTNS_WIDE))
			colorImg = Image.new('RGB', (bg_btn_w, height_per_button), color)
			colorTxt = ','.join(str(x) for x in color)
			
			unselected = colorImg.copy()
			DrawInteriorBorder(unselected, max(1, height_per_button//6), (255, 255, 255))
			
			selected   = unselected.copy()
			DrawInteriorBorder(selected, max(2, height_per_button//5), (255, 255, 255))
			DrawInteriorBorder(selected, max(2, height_per_button//5), (255,   0,   0))
			
			colorBtn = SelectableObject(colorTxt, (xpos, ypos), colorImg.size, selected, unselected)			
			colorBtn.SetState(SelectableObject.UNSELECTED)
			self.colorbtns.append(colorBtn)
								
		self.buttons  = tmp_btns
		self.genGrads = False
		
	def HandleGradientClick(self, mouse, displacedMouse):
		for grad in self.grads:
			if grad.MouseOver(displacedMouse):
				grad.SetState()
				return True
		return False
		
	def HandleButtonClick(self, mouse, displacedMouse):
	
		if not GuiWrapper.HandleButtonClick(self, mouse, displacedMouse):
			if self.contourBtn.MouseOver(mouse):
				print('Creating gradients from selection...')
				self.genGrads = True
				return True
				
			for btn in self.colorbtns:
				if btn.MouseOver(mouse):
					self.SelectBGColor(btn)
					return True
				
		return False
		
	def HandleEvents(self):
	
		mouse = pygame.mouse.get_pos()
		for event in pygame.event.get():
		
			if event.type == pygame.QUIT:
				return False
		
			# Check and see if the mouse clicks one of our buttons.
			if event.type == pygame.MOUSEBUTTONDOWN:
			
				if self.HandleMouseDown(event):
					if self.genGrads:
						return False

		# Highlight buttons our mouse is over
		self.HighlightButtons()
		return True		


	