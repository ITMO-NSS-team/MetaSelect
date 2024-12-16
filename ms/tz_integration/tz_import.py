import sys

from sb.utils.sb_utils import parse_sb_config, make_path


def run():
    _ = parse_sb_config()
    tabzilla_path = "/submodules/tabzilla/TabZilla"
    tabzilla_path = make_path(tabzilla_path)

    sys.path.append(tabzilla_path)

