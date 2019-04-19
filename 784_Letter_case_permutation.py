"""
Given a string S, we can transform every letter individually to be
lowercase or uppercase to create another string.  Return a list of
all possible strings we could create.

Examples:
Input: S = "a1b2"
Output: ["a1b2", "a1B2", "A1b2", "A1B2"]

Input: S = "3z4"
Output: ["3z4", "3Z4"]

Input: S = "12345"
Output: ["12345"]
"""
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        def backtracking(pos, current_exp):
            if pos==len(S):
                self.ans.append(current_exp)
                return
            if S[pos] in '0123456789':
                backtracking(pos+1, current_exp+S[pos])
            else:
                backtracking(pos+1, current_exp+S[pos].lower())
                backtracking(pos+1, current_exp+S[pos].upper())

        self.ans = []
        backtracking(0, '')
        return self.ans
