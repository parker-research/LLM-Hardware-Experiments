def split_large_file_by_large_line_breaks(
    file_contents: str,
    min_lines_per_chunk: int,
    max_lines_per_chunk: int,
) -> str:
    """
    Splits the file_contents into chunks of lines. Tries to split at places where there are several
    consecutive empty lines. The chunks will have at least min_lines_per_chunk lines and at most
    max_lines_per_chunk lines.
    """
    lines = file_contents.replace("\r", "").split("\n")
    locations_with_2_empty_lines = [
        i for i in range(len(lines) - 1) if lines[i].strip() == "" and lines[i + 1].strip() == ""
    ]

    locations_with_3_empty_lines = [
        i
        for i in range(len(lines) - 2)
        if lines[i].strip() == "" and lines[i + 1].strip() == "" and lines[i + 2].strip() == ""
    ]

    # use these places to stitch together the chunks
    locations = locations_with_2_empty_lines + locations_with_3_empty_lines
    locations = sorted(locations)

    chunks = []

    # TODO: this function wasn't really validated

    start = 0
    for location in locations:
        if location - start >= min_lines_per_chunk:
            chunks.append(lines[start:location])
            start = location

    if len(lines) - start >= min_lines_per_chunk:
        chunks.append(lines[start:])
    else:
        chunks[-1].extend(lines[start:])
    return "\n".join(["\n".join(chunk) for chunk in chunks])
