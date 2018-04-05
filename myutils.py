from contextlib import contextmanager
from timeit import default_timer

nesting_level = 0
newl = False
@contextmanager
def msg(message):
	global nesting_level, newl

	indent = '  ' * nesting_level
	nl = '\n' if newl else ''
	output = f'{nl}{indent}{message} ... '
	print(output, end='', flush=True)
	nesting_level += 1
	newl = True

	t0 = default_timer()
	yield
	t1 = default_timer()
	
	print('{}done ({:.3f}s)'.format('' if newl else indent, t1-t0))
	nesting_level -= 1
	newl = False