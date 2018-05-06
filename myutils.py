from contextlib import contextmanager

nesting_level = 0
newl = False
@contextmanager
def msg(message, enabled=True, done=True):
    if enabled is False: 
        yield
        return
    global nesting_level, newl

    indent = '  '
    nl = '\n' if newl else ''
    output = f'{nl}{indent * nesting_level}{message} {"..." if done else ""}'
    print(output, end='', flush=True)
    nesting_level += 1
    newl = True

    from timeit import default_timer
    t0 = default_timer()
    yield
    t1 = default_timer()
    
    if done: print('{}done ({:.3f}s)'.format(' ' if newl else indent * (nesting_level), t1-t0))
    else: print()
    nesting_level -= 1
    newl = False

@contextmanager
def suppress_numpy_err():
    from numpy import seterr
    old_settings = seterr(all='ignore')
    yield
    seterr(**old_settings)