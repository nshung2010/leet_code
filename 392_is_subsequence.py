"""
Given a string s and a string t, check if s is subsequence of t.

You may assume that there is only lower case English letters in both s
and t. t is potentially a very long (length ~= 500,000) string, and s
is a short string (<=100).

A subsequence of a string is a new string which is formed from the
original string by deleting some (can be none) of the characters without
disturbing the relative positions of the remaining characters.
(ie, "ace" is a subsequence of "abcde" while "aec" is not).

Example 1:
s = "abc", t = "ahbgdc"

Return true.

Example 2:
s = "axc", t = "ahbgdc"

Return false.

Follow up:
If there are lots of incoming S, say S1, S2, ... , Sk where k >= 1B,
and you want to check one by one to see if T has its subsequence.
In this scenario, how would you change your code?
"""
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        def helper(ch, t):
            for i, t_ch in enumerate(t):
                # print(t_ch)
                if t_ch==ch:
                    return(i, t[i+1:])
            return (False, t)
        for ch in s:
            # print(ch)
            bool_, t = helper(ch, t)
            # print(bool_, t)
            if bool_ is False:
                return False
        return True
class Solution(object):
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        i = 0
        for c in t:
            if c == s[i]:
                i += 1
                if i == len(s):
                    return True
        return False

class Solution(object):
     def isSubsequence(self, s, t):
        def binary_search(idx_list, tidx, left, right):
            # worst cast t has all same chars, O(logN) N=len(t)
            while left < right:
                mid = left + (right - left) // 2
                if idx_list[mid] > tidx:
                    right = mid
                else:
                    left = mid + 1

            return idx_list[left] if left < len(idx_list) else -1

        if len(s) == 0: return True
        d = collections.defaultdict(list)
        for i, ch in enumerate(t):
            d[ch] += [i]

        tidx = -1
        # M=len(s), O(M*logN)
        for ch in s:
            if ch not in d: return False
            # print(d[ch], tidx)
            tidx = binary_search(d[ch], tidx, 0, len(d[ch]))

            if tidx == -1: return False

        return True
