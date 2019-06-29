# Smartgrad

### What is Smartgrad?

Smartgrad is a tool that converts images into gradients, mainly for use with fractal rendering software like Apophysis or Chaotica. It's still a work-in-progress, but as enough people wanted to see it, I figured I'd upload what I had without caring so much about presentation / style / etc.

Requires `python3` with f-strings to run (i.e 3.6+, I believe?)

### How do I use it?

1. Use `python _createme.py` to run cython and compile these for around a 2x speed boost over pure python code. Namely, this makes the actual gradient generation and extraction faster since that's pure python
2. You can just change the `.pyx` to `.py` if you don't care as much about that.
3. There's a bunch of packages you'll need to run this (numpy+mkl, opencv, pygame, pillow-simd, etc)

To run minimally:
1. `python smartgrad.py -e -i <path_to_input_file>`

Gradients generated will be pasted into your clipboard. Or so I assume, but I've only tested this functionality on Windows.

It can also generate gradients to a file.
2. `python smartgrad.py -i <path_to_input_file> -o <output_gradient_directory>`

There's also oodles of other options you can see by passing in `-h` to the command line.  Not all of them might be applicable, but I haven't done a pass through to clean them all up yet.
