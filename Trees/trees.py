from typing import List


class TreeNode:
    def __init__(self, value):
        self.key = value
        self.left = None
        self.right = None

def printTreeInOrder(root: TreeNode) -> None:
    if root is not None:
        printTreeInOrder(root.left)
        print(root.key)
        printTreeInOrder(root.right)

def printTreePreOrder(root: TreeNode) -> None:
    if root is not None:
        print(root.key)
        printTreePreOrder(root.left)
        printTreePreOrder(root.right)

def printTreePostOrder(root: TreeNode) -> None:
    if root is not None:
        printTreePostOrder(root.left)
        printTreePostOrder(root.right)
        print(root.key)

def generateInOrderTree(root: TreeNode) -> List[int]:
    if root is None:
        return []

    left: List[int] = generateInOrderTree(root.left)
    right: List[int] = generateInOrderTree(root.right)

    return left + [root.key] + right

def insertNode(root: TreeNode, key: int) -> TreeNode:
    if root is None:
         return TreeNode(key)

    if key < root.key:
        root.left = insertNode(root.left, key)
    else: 
        key > root.key
        root.right = insertNode(root.right, key)

    return root

keys: List[int] = [5, 2, 7, 1, 3, 6, 8]

a:TreeNode = None

for key in keys:
    a = insertNode(a, key)

'''
a = TreeNode(5)
b = TreeNode(2)
c = TreeNode(7)
d = TreeNode(1)
e = TreeNode(3)
f = TreeNode(6)
g = TreeNode(8)


a.left = b
a.right = c

insertNode(a, 1)
'''

printTreeInOrder(a)
print()
printTreePreOrder(a)
print()
printTreePostOrder(a)
print()
print(f"InOrder Tree = ", generateInOrderTree(a))


'''
     5
   /   \
  2     7
 / \   / \
1   3 6   8
'''