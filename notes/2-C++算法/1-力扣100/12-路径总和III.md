# **437. 路径总和 III**

给定一个二叉树的根节点 `root` 和一个整数 `targetSum`，求该二叉树里节点值之和等于 `targetSum` 的路径的数目。

路径**不需要从根节点开始**，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

## 算法设计
这道题采用**双重递归**。外层递归枚举每个节点作为路径起点，内层递归从当前起点向下搜索，并不断用目标值减去经过的节点值；当剩余目标值等于当前节点值时，说明找到一条合法路径。最坏时间复杂度为 $O(n^2)$，空间复杂度为 $O(h)$。

## 代码实现
```cpp
class Solution {
public:
    int rootSum(TreeNode *root, long long targetSum)
    {
        if (!root) {
            return 0;
        }
        int res = 0;
        if (root->val == targetSum) {
            res ++;
        }
        res += rootSum(root->left, targetSum- root->val);
        res += rootSum(root->right, targetSum- root->val);
        return res;
    }

    int pathSum(TreeNode* root, int targetSum) {
        if (!root) {
            return 0;
        }
        int res = rootSum(root, targetSum);
        res += pathSum(root->left, targetSum);
        res += pathSum(root->right, targetSum);
        return res;
    }
};
```