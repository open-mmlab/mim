from mim.utils import cast2lowercase


def param2lowercase(ctx, param, value):
    if value is not None:
        return cast2lowercase(value)
