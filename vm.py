"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.offset = 0
        self.position: tp.Any = {}
        self.exception = Exception

    @staticmethod
    def bind_args(code: tp.Any, kwdefaults: tp.Any, defaults: tp.Any,
                  *args: tp.Any, **kwargs: tp.Any) -> dict[str, tp.Any]:
        default_values = defaults or ()
        kwonlydefaults = kwdefaults or {}

        has_varargs = bool(code.co_flags & CO_VARARGS)
        has_varkwargs = bool(code.co_flags & CO_VARKEYWORDS)

        posonly_slice = slice(None, code.co_posonlyargcount)
        pos_or_kw_slice = slice(code.co_posonlyargcount, code.co_argcount)
        kwonly_slice = slice(code.co_argcount, code.co_argcount + code.co_kwonlyargcount)
        defaults_slice = slice(code.co_argcount - len(default_values), code.co_argcount)

        # parse defaults
        default_names = code.co_varnames[defaults_slice]
        defaults = dict(zip(default_names, default_values))

        # parse args
        parsed_posonlyargs = dict(zip(
            code.co_varnames[posonly_slice],
            args[posonly_slice]
        ))
        parsed_posargs = dict(zip(
            code.co_varnames[pos_or_kw_slice],
            args[pos_or_kw_slice]
        ))
        varargs = args[code.co_argcount:]

        # parse kwargs
        posonly_names = frozenset(code.co_varnames[posonly_slice])
        pos_or_kw_names = frozenset(code.co_varnames[pos_or_kw_slice])
        kwonly_names = frozenset(code.co_varnames[kwonly_slice])

        parsed_kwargs = {}
        parsed_kwonlyargs = {}
        varkwargs = {}
        for k, v in kwargs.items():
            if k in pos_or_kw_names:
                parsed_kwargs[k] = v
            elif k in kwonly_names:
                parsed_kwonlyargs[k] = v
            else:
                varkwargs[k] = v

        # checks
        if parsed_posargs.keys() & parsed_kwargs.keys():
            raise TypeError(ERR_MULT_VALUES_FOR_ARG)

        if varkwargs.keys() & posonly_names and not has_varkwargs:
            raise TypeError(ERR_POSONLY_PASSED_AS_KW)

        if varkwargs and not has_varkwargs:
            raise TypeError(ERR_TOO_MANY_KW_ARGS)

        if varargs and not has_varargs:
            raise TypeError(ERR_TOO_MANY_POS_ARGS)

        if (
                (posonly_names | pos_or_kw_names)
                - parsed_posonlyargs.keys()
                - parsed_posargs.keys()
                - parsed_kwargs.keys()
                - defaults.keys()
        ):
            raise TypeError(ERR_MISSING_POS_ARGS)

        if kwonly_names - parsed_kwonlyargs.keys() - kwonlydefaults.keys():
            raise TypeError(ERR_MISSING_KWONLY_ARGS)

        # make result
        bound_args = {}
        bound_args.update(defaults)
        bound_args.update(kwonlydefaults)
        bound_args.update(parsed_posonlyargs)
        bound_args.update(parsed_posargs)
        bound_args.update(parsed_kwargs)
        bound_args.update(parsed_kwonlyargs)

        if has_varargs:
            varargs_name = code.co_varnames[code.co_argcount + code.co_kwonlyargcount]
            bound_args[varargs_name] = varargs

        if has_varkwargs:
            varkwargs_name = code.co_varnames[code.co_argcount + code.co_kwonlyargcount + has_varargs]
            bound_args[varkwargs_name] = varkwargs

        return bound_args

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))
        for x, y in enumerate(instructions):
            self.position[y.offset] = x
        with_arg = ("load_fast_and_clear", "load_const", "load_fast", "store_name", "store_fast", "store_attr",
                    "delete_name", "delete_fast", "delete_attr", "import_name", "import_from",
                    "compare_op", "format_value", "return_const")
        i = 0
        while i < len(instructions):
            instruction = instructions[i]
            if instruction.opname.lower() in with_arg:
                getattr(self, instruction.opname.lower() + "_op")(instruction.arg)
                if (instruction.opname.lower() == "return_const") and (i + 1 != len(instructions)):
                    break
            else:
                getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if self.offset != 0:
                i = self.offset
                self.offset = 0
            else:
                i += 1
        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-CALL
        """
        arguments = self.popn(arg)
        f = self.pop()
        if not callable(f):
            f = self.pop()
        self.push(f(*arguments))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def binary_op_op(self, arg: tp.Any) -> None:
        x, y = self.popn(2)
        if arg == 0:
            self.push(x + y)
        elif arg == 1:
            self.push(x & y)
        elif arg == 2:
            self.push(x // y)
        elif arg == 3:
            self.push(x << y)
        elif arg == 5:
            self.push(x * y)
        elif arg == 6:
            self.push(x % y)
        elif arg == 7:
            self.push(x | y)
        elif arg == 8:
            self.push(x ** y)
        elif arg == 9:
            self.push(x >> y)
        elif arg == 10:
            self.push(x - y)
        elif arg == 11:
            self.push(x / y)
        elif arg == 12:
            self.push(x ^ y)
        elif arg == 13:
            x += y
            self.push(x)
        elif arg == 14:
            x &= y
            self.push(x)
        elif arg == 15:
            x //= y
            self.push(x)
        elif arg == 16:
            x <<= y
            self.push(x)
        elif arg == 18:
            x *= y
            self.push(x)
        elif arg == 19:
            x %= y
            self.push(x)
        elif arg == 20:
            x |= y
            self.push(x)
        elif arg == 21:
            x **= y
            self.push(x)
        elif arg == 22:
            x >>= y
            self.push(x)
        elif arg == 23:
            x -= y
            self.push(x)
        elif arg == 24:
            x /= y
            self.push(x)
        elif arg == 25:
            x ^= y
            self.push(x)

    def unpack_sequence_op(self, arg: int) -> None:
        assert (len(self.top()) == arg)
        self.data_stack.extend(self.pop()[:-arg - 1:-1])

    def binary_slice_op(self, arg: tp.Any) -> None:
        b = self.pop()
        a = self.pop()
        s = self.pop()
        self.push(s[a:b])

    def build_slice_op(self, arg: int) -> None:
        if arg == 2:
            b = self.pop()
            a = self.pop()
            self.push(slice(a, b))
        elif arg == 3:
            i = self.pop()
            b = self.pop()
            a = self.pop()
            self.push(slice(a, b, i))

    def store_slice_op(self, arg: tp.Any) -> None:
        end = self.pop()
        start = self.pop()
        container = self.pop()
        value = self.pop()
        container[start:end] = value

    def binary_subscr_op(self, arg: tp.Any) -> None:
        key = self.pop()
        container = self.pop()
        self.push(container[key])

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(self.code.co_consts[arg])

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def return_const_op(self, arg: tp.Any) -> None:
        self.return_value = self.code.co_consts[arg]

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def build_tuple_op(self, count: int) -> None:
        self.push(tuple(self.popn(count)))

    def build_list_op(self, count: int) -> None:
        self.push(self.popn(count))

    def list_extend_op(self, i: int) -> None:
        seq = self.pop()
        list.extend(self.data_stack[-i], seq)

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        values = []
        for i in range(count):
            values.append(self.pop())
        values = values[::-1]
        d = {}
        for key, value in zip(keys, values):
            d[key] = value
        self.push(d)

    def build_set_op(self, count: int) -> None:
        self.push(set(self.popn(count)))

    def build_map_op(self, count: int) -> None:
        a = self.popn(2 * count)
        d = {}
        for i in range(0, count, 2):
            d[a[i]] = a[i + 1]
        self.push(d)

    def copy_op(self, i: int) -> None:
        assert i > 0
        self.push(self.data_stack[-i])

    def set_update_op(self, i: int) -> None:
        seq = self.pop()
        set.update(self.data_stack[-i], seq)

    def unary_negative_op(self, arg: tp.Any) -> None:
        self.push(-self.pop())

    def unary_not_op(self, arg: tp.Any) -> None:
        self.push(not self.pop())

    def unary_invert_op(self, arg: tp.Any) -> None:
        self.push(~self.pop())

    def unary_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def call_intrinsic_1_op(self, arg: tp.Any) -> None:
        pass

    def load_attr_op(self, arg: str) -> None:
        self.push(getattr(self.pop(), arg))

    def compare_op_op(self, arg: tp.Any) -> None:
        y = self.pop()
        x = self.pop()
        if arg == 40:
            self.push(x == y)
        elif arg == 55:
            self.push(x != y)
        elif arg == 68:
            self.push(x > y)
        elif arg == 92:
            self.push(x >= y)
        elif arg == 2:
            self.push(x < y)
        elif arg == 26:
            self.push(x <= y)
        else:
            raise NotImplementedError

    def swap_op(self, i: int) -> None:
        self.data_stack[-i], self.data_stack[-1] = self.data_stack[-1], self.data_stack[-i]

    def contains_op_op(self, invert: int) -> None:
        y = self.pop()
        x = self.pop()
        if not invert:
            self.push(x in y)
        else:
            self.push(x not in y)

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)

        kw_defaults = {}
        defaults = ()

        if arg & 0x02:
            kw_defaults = self.pop()
        if arg & 0x01:
            defaults = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = self.bind_args(code, kw_defaults, defaults, *args, **kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[self.code.co_names[arg]] = const

    def store_fast_op(self, arg: tp.Any) -> None:
        self.locals[self.code.co_varnames[arg]] = self.pop()

    def load_fast_op(self, arg: tp.Any) -> None:
        i = self.code.co_varnames[arg]
        if i in self.locals:
            self.push(self.locals[i])
        elif i in self.globals:
            self.push(self.globals[i])
        elif i in self.builtins:
            self.push(self.builtins[i])

    def pop_jump_if_true_op(self, delta: int) -> None:
        if self.top():
            self.offset = self.position[delta]
        self.pop()

    def pop_jump_if_false_op(self, delta: int) -> None:
        if not self.top():
            self.offset = self.position[delta]
        self.pop()

    def store_subscr_op(self, arg: tp.Any) -> None:
        key = self.pop()
        container = self.pop()
        value = self.pop()
        container[key] = value

    def delete_subscr_op(self, arg: tp.Any) -> None:
        key = self.pop()
        container = self.pop()
        del container[key]

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def load_fast_and_clear_op(self, arg: tp.Any) -> None:
        i = self.code.co_varnames[arg]
        if i in self.locals:
            self.push(self.locals[i])
        elif i in self.globals:
            self.push(self.globals[i])
        elif i in self.builtins:
            self.push(self.builtins[i])
        else:
            self.push(None)

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        self.push(AssertionError)

    def raise_varargs_op(self, arg: tp.Any) -> None:
        if arg == 0:
            raise self.exception
        elif arg == 1:
            raise self.pop()
        elif arg == 2:
            x, y = self.popn(2)
            raise x from y

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(self.builtins['__build_class__'])

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if '__annotations__' not in self.locals:
            self.locals['__annotations__'] = {}

    def import_name_op(self, arg: tp.Any) -> None:
        imp = self.code.co_names[arg]
        level, fromlist = self.popn(2)
        self.push(__import__(imp, fromlist=fromlist, level=level))

    def import_from_op(self, arg: tp.Any) -> None:
        self.push(getattr(self.top(), self.code.co_names[arg]))

    def delete_fast_op(self, arg: tp.Any) -> None:
        i = self.code.co_varnames[arg]
        if i in self.locals:
            del self.locals[i]

    def kw_names_op(self, arg: tp.Any) -> None:
        pass

    def store_attr_op(self, arg: tp.Any) -> None:
        obj = self.pop()
        value = self.pop()
        i = self.code.co_names[arg]
        setattr(obj, i, value)

    def extended_arg_op(self, arg: tp.Any) -> None:
        pass

    def is_op_op(self, invert: int) -> None:
        y = self.pop()
        x = self.pop()
        if not invert:
            self.push(x is y)
        else:
            self.push(x is not y)

    def pop_jump_if_none_op(self, delta: int) -> None:
        if self.top() is None:
            self.offset = self.position[delta]
        self.pop()

    def jump_forward_op(self, delta: int) -> None:
        self.offset = self.position[delta]

    def jump_backward_op(self, delta: int) -> None:
        self.offset = self.position[delta]

    def end_for_op(self, arg: tp.Any) -> None:
        self.pop()
        self.pop()

    def delete_name_op(self, arg: tp.Any) -> None:
        i = self.code.co_names[arg]
        if i in self.locals:
            del self.locals[i]
        elif i in self.globals:
            del self.globals[i]
        elif i in self.builtins:
            del self.builtins[i]

    def for_iter_op(self, delta: tp.Any) -> None:
        while self.top() is None:
            self.pop()
        try:
            self.push(next(self.top()))
        except StopIteration:
            self.offset = self.position[delta]
            self.push(None)

    def list_append_op(self, i: tp.Any) -> None:
        item = self.pop()
        list.append(self.data_stack[-i], item)

    def delete_global_op(self, arg: str) -> None:
        if arg in self.globals:
            self.globals.pop(arg)
        elif arg in self.builtins:
            self.builtins.pop(arg)
        else:
            raise NameError

    def format_value_op(self, flags: tp.Any) -> None:
        x = ''
        if flags & 0x04:
            x = self.pop()
        if flags & 0x03 == 0:
            self.push(f"{self.pop():{x}}")
        elif flags & 0x03 == 1:
            self.push(f"{self.pop()!s:{x}}")
        elif flags & 0x03 == 2:
            self.push(f"{self.pop()!r:{x}}")
        elif flags & 0x03 == 3:
            self.push(f"{self.pop()!a:{x}}")

    def build_string_op(self, count: int) -> None:
        self.push(''.join(self.popn(count)))

    def delete_attr_op(self, arg: tp.Any) -> None:
        self.pop()

    def set_add_op(self, i: tp.Any) -> None:
        item = self.pop()
        set.add(self.data_stack[-i], item)

    def map_add_op(self, i: tp.Any) -> None:
        value = self.pop()
        key = self.pop()
        dict.__setitem__(self.data_stack[-i], key, value)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
