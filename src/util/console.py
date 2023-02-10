

VALUE_CHARS = " .:+*#"


def print_2d(state, threshold: float = .5, file=None):
    max_v = 0.000001
    for row in state:
        for v in row:
            max_v = max(max_v, v)

    num_chars = len(VALUE_CHARS)
    for row in state:
        print("".join(
            VALUE_CHARS[max(0, min(num_chars - 1, int(v / max_v * num_chars + .5)))]
            for v in row
        ), file=file)
