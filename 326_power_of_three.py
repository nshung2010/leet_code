"""
Given an integer, write a function to determine if it is a power of three.

Example 1:

Input: 27
Output: true
"""
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n==0:
            return False
        i = 0
        while 3**i < n:
            i += 1
        if n==3**i:
            return True
        return False
