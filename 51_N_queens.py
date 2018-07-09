class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.size = n
        self.build_board()
        res = []
        self.solve(res)
        return res
  
    def build_board(self):
        n = self.size
        self.board = ["."*n for _ in range(n)]
        return None
    
    def set_Q(self, i, j, char):
        s = self.board[i]
        n = self.size
        self.board[i] = s[0:j] + char + s[j+1:n]
        
    def is_safe_row(self,row):
        n = self.size
        for i in range(n):
            s = self.board[row]
            if self.board[row][i] == 'Q':
                return False
        return True
    
    def is_safe_col(self,col):
        n = self.size
        for i in range(n):
            if self.board[i][col] == 'Q':
                return False
        return True
    
    def is_safe_diagonal(self,row, col):
        n = self.size
        for i in range(n):
            j1 = i-row+col
            j2 = row+col - i
            if 0<=j1<n and self.board[i][j1] == 'Q':
                return False
            if 0<=j2<n and self.board[i][j2] == 'Q':
                return False
        return True
    
    def is_safe(self, row, col):
        return self.board[row][col] != 'Q' and self.is_safe_row(row) and self.is_safe_col(col) and self.is_safe_diagonal(row, col)
    
    def get_number_of_queens(self):
        count = 0
        n = self.size
        for row in range(n):
            for col in range(n):
                if self.board[row][col] == 'Q':
                    count += 1
        return count
    def find_safe_row(self):
        n = self.size
        for i in range(n):
            if self.is_safe_row(i):
                return i
        return None
    
    def find_safe_col(self, row):
        n = self.size
        for col in range(n):
            if self.is_safe(row,col):
                return col
        return None
        
    def solve(self, res):
        n_queens = self.get_number_of_queens()
        n = self.size
        if n_queens == n:
            res.append(self.board)
        for i in range(n):
            if self.is_safe_row(i):    
                #print(i)
                j = self.find_safe_col(i)
                if j is not None:
                    #print('find j', j)
                    self.set_Q(i,j,'Q')
                    #print(self.board)
                    if self.solve(res):
                        self.set_Q(i,j,'.')
                        return True
                    self.set_Q(i,j,'.')
        return False
       
                    
    
    

class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.size = n
        res = []
        self.reset_board()
        self.solve(res, 0, n)
        return res
  
    def reset_board(self):
        n = self.size
        self.board = ["."*n for _ in range(n)]
        return None
    
    def set_Q(self, i, j, char):
        s = self.board[i]
        n = self.size
        self.board[i] = s[0:j] + char + s[j+1:n]
        
    def is_safe_row(self,row):
        n = self.size
        for i in range(n):
            s = self.board[row]
            if self.board[row][i] == 'Q':
                return False
        return True
    
    def is_safe_col(self,col):
        n = self.size
        for i in range(n):
            if self.board[i][col] == 'Q':
                return False
        return True
    
    def is_safe_diagonal(self,row, col):
        n = self.size
        for i in range(n):
            j1 = i-row+col
            j2 = row+col - i
            if 0<=j1<n and self.board[i][j1] == 'Q':
                return False
            if 0<=j2<n and self.board[i][j2] == 'Q':
                return False
        return True
    
    def is_safe(self, row, col):
        return self.board[row][col] != 'Q' and self.is_safe_row(row) and self.is_safe_col(col) and self.is_safe_diagonal(row, col)
    
    def solve(self, res, row, n):
        if row >= n:
            return
        for i in range(n):
            if self.is_safe(row, i):    
                self.set_Q(row,i,'Q')
                if row == n-1:
                    print(self.board)
                    res.append(self.board)
                    self.set_Q(row, i, '.')
                    return
                self.solve(res, row +1, n)
                self.set_Q(row, i,'.')
      



class Solution(object):
    def solveNQueens(self, n):
        res = []
        for i in range(n):
            self._solve(res, [i], n)
        return res
    
    def _solve(self, res, cols, n):
        m = len(cols)
        if m == n:
            res.append(self._make_grid(cols))
        else:
            for i in range(n):
                for j, col in enumerate(cols):
                    if abs(i-col) == m-j or i==col:
                        break
                else:
                    cols.append(i)
                    self._solve(res, cols, n)
                    cols.pop()
    
    def _make_grid(self, cols):
        n = len(cols)
        return ["".join(["Q" if i == col else "." for i in range(n)]) for col in cols]
            

class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.count = 0
        for i in range(n):
            self._solve([i], n)
        return self.count
    
    def _solve(self, cols, n):
        m = len(cols)
        if m == n:
            self.count += 1
            print(self.count)
        else:
            for i in range(n):
                for j, col in enumerate(cols):
                    if abs(i-col) == m-j or i==col:
                        break
                else:
                    cols.append(i)
                    self._solve(cols, n)
                    cols.pop()