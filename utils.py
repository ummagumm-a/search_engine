def get_shape(tpl):
    return list(map(lambda x: x.shape[1], tpl))

def split_vec(vec, splits):
    return (vec[:splits[0]],
            vec[splits[0]:splits[0] + splits[1]],
            vec[splits[0] + splits[1]:])
