# 467 Unique Substrign in Wraparound String
"""
Consider the string s to be the infinite wraparound string of
"abcdefghijklmnopqrstuvwxyz", so s will look like this:
"...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".

Now we have another string p. Your job is to find out how many unique
non-empty substrings of p are present in s. In particular, your input
is the string p and you need to output the number of different non-empty
substrings of p in the string s.

Note: p consists of only lowercase English letters and the size of p
might be over 10000.

Example 1:
Input: "a"
Output: 1

Explanation: Only the substring "a" of string "a" is in the string s.
Example 2:
Input: "cac"
Output: 2
Explanation: There are two substrings "a", "c" of string "cac" in the
string s.
Example 3:
Input: "zab"
Output: 6
Explanation: There are six substrings "z", "a", "b", "za", "ab", "zab"
 of string "zab" in the string s.
"""
class Solution(object):
    def findSubstringInWraproundString(self, p):
        """
        :type p: str
        :rtype: int
        """
        sequence_len = 0
        max_len_end_at = {}
        count = 0
        for i, c in enumerate(p):
            if i>0 and ord(p[i])-ord(p[i-1]) in [1, -25]:
                sequence_len += 1
            else:
                sequence_len = 1
            if c not in max_len_end_at:
                count += sequence_len
                max_len_end_at[c] = sequence_len
            elif sequence_len > max_len_end_at[c]:
                count += sequence_len - max_len_end_at[c]
                max_len_end_at[c] = sequence_len
            else:
                pass

        return count

class Solution(object):
    def findSubstringInWraproundString(self, p):
        """
        :type p: str
        :rtype: int
        """
        prev = -1
        count = 0
        dicts = [0 for i in xrange(26)]
        for i in p:
            idx = ord(i)-97
            if prev == -1: count = 1
            elif (idx - prev)%26 == 1: count += 1
            else: count = 1
            prev = idx
            dicts[idx] = max(dicts[idx], count)

        return sum(dicts)
