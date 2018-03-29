from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def msg(message, newline=False):
	if newline:
		print(message + ' ... ', flush=True)
	else:
		print(message + ' ... ', end='', flush=True)
	t0 = default_timer()
	yield
	t1 = default_timer()
	print("done ({:.3f}s)".format(t1-t0))