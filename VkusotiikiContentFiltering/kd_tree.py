from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Tree(namedtuple('Tree', 'root left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


def kdtree(point_list, depth=0):

    if not len(point_list):
        return None

    k = len(point_list[0])

    # Select attribute based on depth
    attr = depth % k

    # Sort point list and choose median as tree root
    point_list.sort(key=itemgetter(attr))

    median = len(point_list) // 2  # choose median

    # Create tree and construct subtrees
    return Tree(
        root=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )


def main():
    """Example usage"""
    point_list = [(2, 3), (4, 7), (5, 4), (7, 2), (8, 1), (9, 6)]
    tree = kdtree(point_list)
    print(tree)


if __name__ == '__main__':
    main()
