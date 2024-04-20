
def make_header_str(label: str, width: int = 80, char: str = '=') -> str:
    """Return a string with a header label centered in a line of width characters.
    E.g., make_header_str('Start Prompt', char='>', width=54) returns '>>>>>>>>>>>>>>>>>>>> Start Prompt >>>>>>>>>>>>>>>>>>>>'
    """
    assert len(char) == 1, "char must be a single character"
    assert len(label) < width - 2, f"label '{label}' is too long for width {width}"
    
    len_before = (width - len(label) - 2) // 2
    len_after = len_before if (len_before * 2 + len(label) + 2 == width) else len_before + 1
    header_str = f"{char * len_before} {label} {char * len_after}"
    assert len(header_str) == width, f"len(header_str) = {len(header_str)} != {width}"
    return header_str
