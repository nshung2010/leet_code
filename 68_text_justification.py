class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        i = 0
        n_char = 0
        start = 0
        end = 0
        res = []
        while i<len(words):
            n_char += len(words[i])+1
            if n_char-1 > maxWidth:
                end = i
                res.append(self.joint_word(words[start:end], maxWidth))
                start=i
                n_char = 0
                i -= 1
            if i==len(words)-1:
                end = i+1
                res.append(self.joint_word(words[start:end], maxWidth, True))
            i += 1
        return res
            
    def joint_word(self, words, maxWidth, b_last_line = False):
        print(words)
        if words == []:
            return ' '*maxWidth
        n_word = len(words)
        n_char = 0
        for word in words:
            n_char += len(word)+1 #add a space after each word
        n_char -= 1 # remove the last space after the last word
        if n_char > maxWidth:
            return False
        if n_word == 1:
            return words[0]+' '*(maxWidth-n_char)
        res = ''
        if not b_last_line:
            n_space_per_word = (maxWidth-n_char)//(n_word-1)
            n_space_left = (maxWidth-n_char)%(n_word-1)
            
            for i in range(n_word-1):
                if i<=n_space_left-1:
                    space_add = ' '*(n_space_per_word+2) #add two extra space, one for the 
                    #default space and another one for unevenly distributed space
                else:
                    space_add = ' '*(n_space_per_word+1) #add one default space only
                res += words[i]+space_add
            res += words[-1]
        else:
            for word in words[0:-1]:
                res += word + ' '
            res += words[-1]   # add the last word
            res += ' '*(maxWidth-n_char)   # add all extra space
        return res

words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
sol = Solution()
print(sol.fullJustify(words, maxWidth))