class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if board == []:
            return False
        self.board = board
        m = len(board)
        n = len(board[0])
        self.checked = {}
        for row in range(m):
            for col in range(n):
                if board[row][col] == word[0]:
                    if self.check_neighbor(word, row, col):
                        return True
        return False
    
    def check_neighbor(self, word, start_row, start_col):
        if word == '':
            return True
        if start_row < 0 or start_row > len(self.board)-1 or start_col < 0 or start_col > len(self.board[0])-1:
            return False
        # print(start_row, start_col, word)
        if self.board[start_row][start_col] != word[0]:
            return False
        elif self.checked == {} or ((start_row, start_col) not in self.checked.keys()):
            self.checked[(start_row, start_col)] = True
            return self.check_neighbor(word[1:],start_row-1, start_col) or self.check_neighbor(word[1:],start_row+1, start_col)  \
        or self.check_neighbor(word[1:],start_row, start_col+1) or self.check_neighbor(word[1:],start_row, start_col-1)

sol = Solution()
board = [["a","a"]]
word = 'aaa'
sol.exist(board, word)