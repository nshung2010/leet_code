"""
Given a set of distinct positive integers, find the largest subset such
that every pair (Si, Sj) of elements in this subset satisfies:

Si % Sj = 0 or Sj % Si = 0.

If there are multiple solutions, return any subset is fine.

Example 1:

Input: [1,2,3]
Output: [1,2] (of course, [1,3] will also be ok)
Example 2:

Input: [1,2,4,8]
Output: [1,2,4,8]
"""
class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        nums=sorted(nums)
        dp=[[nums[i]] for i in range(len(nums))]
        max_len,res=1,dp[0]
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i]%nums[j]==0 and len(dp[i])<len(dp[j])+1:
                    dp[i]=dp[j]+[nums[i]]
                    if len(dp[i])>max_len:
                        max_len,res=len(dp[i]),dp[i]
        return res

# Another solution:

class Solution(object):
    def recur(path,lastIdx):
            for i in range(lastIdx+1,n):
                curr = nums[i]
                if curr % path[-1] == 0 :
                    if curr not in seen or seen[curr] < len(path):
                        seen[curr] = len(path)
                        recur(path + [curr],i)
            d[len(path)] = path

        n = len(nums)
        if n <= 1: return nums

        d = {};seen = {}
        nums.sort()
        for i in range(n-1):
            recur([nums[i]],i)
        maxKey = max(d.keys())
        return d[maxKey]
class Node(object):
        def __init__(self,val,parent):
            self.val=val
            self.children=[]
            self.depth=0
            self.parent=parent
        def append(self,val):
            found=False
            for child in self.children:
                if(val%child.val==0):
                    child.append(val)
                    found=True
            if(not found):
                self.children.append(Node(val,self))
                self.update_depth(1)
        def get_longest_path(self,path):
            path.append(self.val)
            if(len(self.children)==0):
                return path
            maximum_depth=-1
            max_index=-1
            for i in range(0,len(self.children)):
                child_depth=self.children[i].depth
                if(child_depth>maximum_depth):
                    maximum_depth=child_depth
                    max_index=i
            self.children[max_index].get_longest_path(path)
            return path
        def update_depth(self,depth):
            if(depth<self.depth):
                return
            else:
                self.depth=depth
                if(self.parent is not None):
                    self.parent.update_depth(depth+1)
            return

class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if(nums is None):
            return []
        n=len(nums)
        if(n<=1):
            return nums
        nums.sort()
        res=[]
        if(nums[0]==0): #zero could be part of any solution
            res.append(0)
            del nums[0]
        if(nums[0]==1): #to build the tree we need a root node with value 1, here we take care of the fact if it is in the input or not
            res.append(1)
            del nums[0]
        root=Node(1,None)
        for num in nums:
            root.append(num)
        longest_path=root.get_longest_path([])
        del longest_path[0] #delete the root node of 1 at the start of the path
        res+=longest_path
        return res
