def best_responses(player_index, c, delta):
    if player_index in [0, 1, 3, 5, 7, 13, 15, 16, 17, 19]:
        return best_responses_to_d0_d1_d3_d5_d7_d13_d15_d16_d17_d19(c, delta)
    if player_index == 2:
        return best_responses_to_d2(c, delta)
    if player_index in [4, 8, 12]:
        return best_responses_to_d4_d8_d12(c, delta)
    if player_index == 6:
        return best_responses_to_d6(c, delta)
    if player_index in [9, 11]:
        return best_responses_to_d9_d11(c, delta)
    if player_index == 10:
        return best_responses_to_d10(c, delta)
    if player_index == 14:
        return best_responses_to_d14(c, delta)
    if player_index == 18:
        return best_responses_to_d18(c, delta)
    if player_index in [20, 21, 22, 23, 28, 29, 30, 31]:
        return best_responses_to_d20_d21_d22_d23_d28_d29_d30_d31(c, delta)
    if player_index == 24:
        return best_responses_to_d24(c, delta)
    if player_index == 25:
        return best_responses_to_d25(c, delta)
    if player_index == 26:
        return best_responses_to_d26(c, delta)
    if player_index == 27:
        return best_responses_to_d27(c, delta)


def best_responses_to_d0_d1_d3_d5_d7_d13_d15_d16_d17_d19(c, delta):
    return [0, 4, 8, 12]


def best_responses_to_d2(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta:
        list_of_brs1 = [0, 2, 4, 6, 8, 10, 12, 14]
    if c <= delta:
        list_of_brs2 = [18, 19, 26, 27]
    if c == delta:
        return list_of_brs1 + list_of_brs2 + [1, 3, 9, 11, 16, 17, 24, 25]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d4_d8_d12(c, delta):
    return [0, 2, 4, 6, 8, 10, 12, 14]


def best_responses_to_d6(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c <= delta / (1 - delta):
        list_of_brs1 = [16, 17, 24, 25]
    if c >= delta / (1 - delta):
        list_of_brs2 = [0, 2, 4, 6, 8, 10, 12, 14]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d9_d11(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta / (delta + 1):
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta / (delta + 1):
        list_of_brs2 = [9, 11, 13, 15]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d10(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta:
        list_of_brs1 = [0, 2, 4, 6, 8, 10, 12, 14]
    if c <= delta:
        list_of_brs2 = [28, 29, 30, 31]
    if c == delta:
        return [i for i in range(32)]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d14(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c <= delta / (1 - delta):
        list_of_brs1 = [16, 17, 24, 25]
    if c >= delta / (1 - delta):
        list_of_brs2 = [0, 2, 4, 6, 8, 10, 12, 14]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d18(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta:
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta:
        list_of_brs2 = [2, 3, 10, 11]
    if c == delta:
        return list_of_brs1 + list_of_brs2 + [1, 9]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d20_d21_d22_d23_d28_d29_d30_d31(c, delta):
    return [0, 1, 4, 5, 8, 9, 12, 13]


def best_responses_to_d24(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta:
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta:
        list_of_brs2 = [24, 25, 26, 27, 28, 29, 30, 31]
    if c == delta:
        return list_of_brs1 + list_of_brs2 + [16, 20]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d25(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta / (1 + delta):
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta / (1 + delta):
        list_of_brs2 = [24, 25, 26, 27, 28, 29, 30, 31]
    return list_of_brs1 + list_of_brs2


def best_responses_to_d26(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta:
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta:
        list_of_brs2 = [24, 25, 26, 27, 28, 29, 30, 31]
    if c == delta:
        return [i for i in range(32)]
    return list_of_brs1 + list_of_brs2

def best_responses_to_d27(c, delta):
    list_of_brs1, list_of_brs2 = [], []
    if c >= delta  / (1 + delta):
        list_of_brs1 = [0, 4, 8, 12]
    if c <= delta  / (1 + delta):
        list_of_brs2 = [24, 25, 26, 27, 28, 29, 30, 31]
    return list_of_brs1 + list_of_brs2