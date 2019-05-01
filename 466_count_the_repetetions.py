# 466 Count the rêptitions
"""
Define S = [s,n] as the string S which consists of n connected strings s.
For example, ["abc", 3] ="abcabcabc".

On the other hand, we define that string s1 can be obtained from string
s2 if we can remove some characters from s2 such that it becomes s1.
For example, “abc” can be obtained from “abdbec” based on our definition,
but it can not be obtained from “acbbe”.

You are given two non-empty strings s1 and s2 (each at most 100 characters
long) and two integers 0 ≤ n1 ≤ 106 and 1 ≤ n2 ≤ 106. Now consider the s
trings S1 and S2, where S1=[s1,n1] and S2=[s2,n2]. Find the maximum
integer M such that [S2,M] can be obtained from S1.

Example:

Input:
s1="acb", n1=4
s2="ab", n2=2

Return:
2
"""
class Solution(object):
    def getMaxRepetitions(self, s1, n1, s2, n2):
        """
        :type s1: str
        :type n1: int
        :type s2: str
        :type n2: int
        :rtype: int
        """
        count = 0
        index = 0
        indexr = [0] *(len(s2)+1)
        countr = [0] *(len(s2)+1)

        for i in range(0, n1):
            for j in range(0, len(s1)):

                if s1[j] == s2[index]:
                    index += 1
                if index == len(s2):
                    index = 0
                    count += 1
            countr[i] = count
            indexr[i] = index
            for k in range(0, i):
                if index == indexr[k]:
                    prev_count = countr[k]
                    pattern_count = (countr[i]-countr[k])*((n1-1-k)//(i-k))
                    remain_count = countr[k + (n1-1-k)% (i-k)] - countr[k]
                    # print(k, i, countr, indexr, count, index)
                    return (prev_count + pattern_count + remain_count) // n2
        return countr[n1-1]//n2
