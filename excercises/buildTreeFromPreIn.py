from typing import List


class TreeNode:
    def __init__(self, key):
        self.data: int = key
        self.left: TreeNode = None
        self.right: TreeNode = None


'''
Given a preorder and an inorder traversal list, construct the BST

Example:
input: preorder = [3, 9, 20, 15, 7], inorder = [9, 3, 15, 20, 7]
output: [3, 9, 20, null, null, 15, 7]
'''


# ** Optimize for Time complexity using hash map and arrays indexes
def build_tree(preorder: List[int], inorder: List[int]) -> TreeNode:
    if not preorder or not inorder:
        return None

    inorder_dict: dict = dict()
    root: TreeNode

    # Convert lists to map
    for index, value in enumerate(inorder):
        inorder_dict[value] = index

    root = search_tree(preorder, inorder, inorder_dict)

    return root


def search_tree(preorder: List[int],
                inorder: List[int],
                inorder_dict: dict) -> TreeNode:

    if not preorder or not inorder:
        return None

    root: TreeNode = TreeNode(preorder[0])
    mid = inorder_dict[root.data]
    root.left = search_tree(preorder[1:mid+1], inorder[:mid], inorder_dict)
    root.right = search_tree(preorder[mid+1:], inorder[mid+1:], inorder_dict)

    return root


def print_pre_order(root: TreeNode) -> None:
    if root is not None:
        print(root.data)
        print_pre_order(root.left)
        print_pre_order(root.right)


a: TreeNode = TreeNode(3)
b: TreeNode = TreeNode(9)
c: TreeNode = TreeNode(20)
d: TreeNode = TreeNode(15)
e: TreeNode = TreeNode(7)

a.left = b
a.right = c
c.left = d
c.right = e

preorder: List[int] = [3, 9, 20, 15, 7]
inorder: List[int] = [9, 3, 15, 20, 7]

tree: TreeNode = build_tree(preorder, inorder)
print_pre_order(tree)
