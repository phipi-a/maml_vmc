import re
import haiku as hk


def get_hk_regex_keys(model_params: dict, relist: list[str]) -> list[str]:
    def check_regex_list(regex_list, key):
        for regex in regex_list:
            if re.match("^" + regex + "$", key):
                return True
        return False

    l = [l[0] + "." + l[1] for l in hk.data_structures.traverse(model_params)]

    out = [l for l in l if check_regex_list(relist, l)]

    return out
