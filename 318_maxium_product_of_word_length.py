"""
Given a string array words, find the maximum value of
length(word[i]) * length(word[j]) where the two words do not share
common letters. You may assume that each word will contain only lower
case letters. If no such two words exist, return 0.

Example 1:

Input: ["abcw","baz","foo","bar","xtfn","abcdef"]
Output: 16
Explanation: The two words can be "abcw", "xtfn".
Example 2:

Input: ["a","ab","abc","d","cd","bcd","abcd"]
Output: 4
Explanation: The two words can be "ab", "cd".
Example 3:

Input: ["a","aa","aaa","aaaa"]
Output: 0
Explanation: No such pair of words.
"""
class Solution(object):
    # Naive solution O(N^2)
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        max_length = 0
        signature = {w:set(w) for w in words}
        N = len(words)
        for i in range(N):
            for j in range(i+1,N):
                if bool(signature[words[i]] & signature[words[j]]) == False:
                    max_length = max(max_length, len(words[i])*len(words[j]))
        return max_length
    # To improve a solution, we need to find a better way to check
    # if two words share the same common characters.
    # We will use bit-wise manipulation? int32 is 32 bits.
    # There are 26 letters. Set a bit for every character.
    # How do you test if two words have no similar letters? Just AND them.
    # Testing them now becomes a constant time operation
    def sign(self, word):
        value = 0
        for ch in s:
            value = value|(1<<(ord(c)-97))
        return value

    def max_product(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        signature = [self.sign(x) for x in words]
        max_product, N = 0, len(words)
        for i in range(N):
            for j in range(i+1, N):
                if signature[i] & signature[j] == 0:
                    max_product = max(max_product, len(words[i])*len(words[j]))
        return max_product


