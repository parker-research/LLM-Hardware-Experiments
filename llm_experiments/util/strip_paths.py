import re

# All folder paths before 'STRIP_PATHS_START_FOLDER_NAME' variable are replaced with the string
# "/project" in strip_paths_from_text(...).
STRIP_PATHS_START_FOLDER_NAME = "project"


def strip_paths_from_text(text_blob: str) -> str:
    """Replace absolute paths with relative paths in a tool's output."""
    # Replace all paths before STRIP_PATHS_START_FOLDER_NAME with "/project"
    return re.sub(
        # r"(/[^/]+)+/" + STRIP_PATHS_START_FOLDER_NAME + r"/",
        r"([\\/][^\\/\"();\n]+)+[\\/]" + STRIP_PATHS_START_FOLDER_NAME + r"[\\/]",
        "/project/",
        text_blob,
    )
