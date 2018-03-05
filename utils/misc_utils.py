import sys
import time
import copy
import codecs


def print_parametes(name, params):
    """
    Print parameters
    :param name:    parameter's title name
    :param params:  parameters
    :return:
    """
    print('\n' + name + ':')
    for k, v in params.items():
        print('  {k}: {v}'.format(k=k, v=v))
    print()


def parse_params(params, default_params):
    """
    Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("%s is not a valid model parameter" % key)
        # Param is a dictionary
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


def print_time(s, start_time):
    """
    Take a start time, print elapsed duration, and return a new time.
    """
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True):
    """
    Similar to print but with support to flush and output to a file.
    """
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def get_log_f(log_file):
    """
    Get log file stream
    :param log_file:  log file name
    :return:
    """
    log_f = codecs.open(log_file, mode="ab")
    print_out("# log_file=%s" % log_file, log_f)
    return log_f
