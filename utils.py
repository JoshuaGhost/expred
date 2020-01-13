import pickle
import os

NEG = 0
POS = 1

def cache_decorator(*dump_fnames):
    def excution_decorator(func):
        def wrapper(*args, **kwargs):
            if len(dump_fnames) == 1:
                dump_fname = dump_fnames[0]
                if not os.path.isfile(dump_fname):
                    ret = func(*args, **kwargs)
                    with open(dump_fname, 'wb') as fdump:
                        pickle.dump(ret, fdump)
                    return ret
                
                with open(dump_fname, 'rb') as fdump:
                    ret = pickle.load(fdump)
                return ret
            
            rets = None
            for fname in dump_fnames:
                if not os.path.isfile(fname):
                    rets = func(*args, **kwargs)
                    break
            if rets is not None:
                for r, fname in zip(rets, dump_fnames):
                    with open(fname, 'wb') as fdump:
                        pickle.dump(r, fdump)
                return rets
            
            rets = []
            for fname in dump_fnames:
                with open(fname, 'rb') as fdump:
                    rets.append(pickle.load(fdump))
            return tuple(rets)
        return wrapper
    return excution_decorator
