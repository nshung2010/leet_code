"""
Given two arrays of length m and n with digits 0-9 representing two
numbers. Create the maximum number of length k <= m + n from digits
of the two. The relative order of the digits from the same array must
be preserved. Return an array of the k digits.

Note: You should try to optimize your time and space complexity.

Example 1:

Input:
nums1 = [3, 4, 6, 5]
nums2 = [9, 1, 2, 5, 8, 3]
k = 5
Output:
[9, 8, 6, 5, 3]
Example 2:

Input:
nums1 = [6, 7]
nums2 = [6, 0, 4]
k = 5
Output:
[6, 7, 6, 0, 4]
Example 3:

Input:
nums1 = [3, 9]
nums2 = [8, 9]
k = 3
Output:
[9, 8, 9]
"""
# Easy to understand solution
class Solution_1(object):
    def maxNumber(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = [0]*k
        m=len(nums1);n=len(nums2)
        for i in range(0,k+1):
            j=k-i
            if i<=m and j<=n:
                res1 = self.getMaxNumber(nums1, i)
                res2 = self.getMaxNumber(nums2, j)
                merged = self.MergeMaxNums(res1, res2,k)
                res = max(merged, res,k)
        return res

    def MergeMaxNums(self, n1, n2,nk):
        r = []
        while (n1 or n2) and nk>0:
            if n1>n2:
                r.append(n1[0])
                n1=n1[1:]
            else:
                r.append(n2[0])
                n2=n2[1:]
            nk-=1
        return r

    def getMaxNumber(self, nums, L):
        stack = []
        i=0
        while i<len(nums):
            remain = len(nums)-i
            while len(stack) and nums[i]>stack[-1] and L<remain:
                L+=1
                stack.pop()
            L-=1
            stack.append(nums[i])
            i+=1
        return stack

# Better solution

class Solution_2(object):
    def maxNumber(self, nums1, nums2, k):

        def prep(nums, k):
            drop = len(nums) - k
            out = []
            for num in nums:
                while drop and out and out[-1] < num:
                    out.pop()
                    drop -= 1
                out.append(num)
            return out[:k]

        def merge(a, b):
            return [max(a, b).pop(0) for _ in a+b]

        return max(merge(prep(nums1, i), prep(nums2, k-i))
                   for i in range(k+1)
                   if i <= len(nums1) and k-i <= len(nums2))
