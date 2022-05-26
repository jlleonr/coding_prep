from distutils.command.build import build
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
def buildTree(preorder: List[int], inorder: List[int]) -> TreeNode:

    if not preorder or not inorder:
        return None

    root: TreeNode = TreeNode(preorder[0])
    mid = inorder.index(root.data)
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])

    return root

def printPreOrder(root: TreeNode) -> None:
    if root is not None:
        print(root.data)
        printPreOrder(root.left)
        printPreOrder(root.right)

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

tree: TreeNode = buildTree(preorder, inorder)
printPreOrder(tree)
