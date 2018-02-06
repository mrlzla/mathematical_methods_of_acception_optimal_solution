from collections import OrderedDict


class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


def make_table(timeline, values):
    table = LastUpdatedOrderedDict()
    for i, val in enumerate(values):
        table["{}-{}".format(timeline[i][0], timeline[i][1])] = {
            'start_point': timeline[i][0],
            'end_point': timeline[i][1],
            'value': val
        }
    return table
