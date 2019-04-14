"""
Given a string which contains only lowercase letters, remove duplicate
letters so that every letter appear once and only once. You must make
sure your result is the smallest in lexicographical order among all
possible results.

Example 1:

Input: "bcabc"
Output: "abc"
Example 2:

Input: "cbacdcbc"
Output: "acdb"
"""
class Solution(object):
    # Solution 1 is based on stack. The idea is that if we encounter
    # a character, we will remove athe character in the stack if
    # 1) that character is bigger than current character
    # 2) There is the same character that will apear later (i.e. counter
    # of that character >0)
    #
    def remove_duplicate_letters_sol1(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        count = collections.Counter(s)
        used = set()
        for ch in s:
            count[ch] -= 1
            if ch in used:
                continue
            # print(stack, used)
            while stack and count.get(stack[-1], 0)>0 and stack[-1] > ch:
                used.remove(stack.pop())
            stack.append(ch)
            used.add(ch)
        return ''.join(stack)

    # Solution 2 is based on the observation that:
    # Given the string s, the greedy choice (i.e., the leftmost letter
    # in the answer) is the smallest s[i], such that the suffix s[i .. ]
    # contains all the unique letters. (Note that, when there are more
    # than one smallest s[i]'s, we choose the leftmost one. Why?
    # Simply consider the example: "abcacb".)
    # After determining the greedy choice s[i], we get a new string s'
    # from s by removing all letters to the left of s[i],
    # removing all s[i]'s from s.
    # We then recursively solve the problem w.r.t. s'.

    # The runtime is O(26 * n) = O(n).
    def remove_duplicate_letters_sol2(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s: return ''
        count = [0] * 26
        pos = 0 # the position for the smallest s[i]
        for ch in s:
            count[ord(ch)-ord('a')] += 1
        for i, ch in enumerate(s):
            count[ord(ch)-ord('a')] -= 1
            if s[i] < s[pos]:
                pos = i
            if count[ord(ch)-ord('a')] == 0: break
        substring = s[pos+1:]
        substring = substring.replace(s[pos], "")
        return s[pos] + self.remove_duplicate_letters_sol2(substring)
