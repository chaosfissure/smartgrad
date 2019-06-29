import gzip, pickle
from collections import defaultdict
from contextlib import contextmanager

@contextmanager
def ProfileSection():
	import cProfile
	p = cProfile.Profile() # builtins=False?
	p.enable()
	yield
	p.disable()
	p.print_stats('tottime')
	
@contextmanager
def Pyinstrument():
	from pyinstrument import Profiler
	profiler = Profiler()
	profiler.start()
	yield
	profiler.stop()
	print(profiler.output_text(unicode=True, color=False))

	
def Compress_And_Pickle(what):
	return gzip.compress(pickle.dumps(what, pickle.HIGHEST_PROTOCOL))
		
def Decompress_And_Unpickle(what):
	return pickle.loads(gzip.decompress(what))
	
def WritePickleFile(fname, what):
	with open(fname, 'wb') as f:
		f.write(Compress_And_Pickle(what))
		
def ReadPickleFile(fname):
	with open(fname, 'rb') as f:
		return Decompress_And_Unpickle(f.read())

def Chunk(l, chunks):
	chunks = max(1, chunks)
	i = 0
	while i < len(l):
		yield l[i:i+chunks]
		i += chunks
		
def BreakInto(l, sz):
	chunks = [[] for _ in range(sz)]
	loc = 0
	for elem in l:
		chunks[loc % sz].append(elem)
		loc += 1
	return chunks

	
class KeyableDefaultDict(defaultdict):
	def __missing__(self, key):
		self[key] = obj = self.default_factory(key)
		return obj