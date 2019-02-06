# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:39:53 2019

@author: Erwan
"""

from abc import ABC, abstractmethod
import time
import random

class Player(ABC):
	"""
	Abstract class representing a Reversi player
	"""
		
	def __init__(self, human):
		self.human = human
		self._color = 0
		self.name = None
		
	@property
	def color(self):
		return self._color
	
	@color.setter
	def color(self, value):
		if value == 1:
			self.name = 'WHITE'
		elif value == -1:
			self.name = 'BLACK'
		else:
			value = 0
			
		self._color = value
			
	
	@abstractmethod
	def play(self, board):
		pass
	
	def _possible_moves(self, board):
		
		cols = board.columns
		rows = board.rows
		
		def valid_line(board, x, y, dx, dy):
			"""
			Check a given line to see if a move is valid.
			"""
			if not 0 <= x + dx + dx < 8:
				return False
			if not 0 <= y + dy + dy < 8:
				return False
			
			coords_1 = board.columns[x+dx] + board.rows[y+dy]
			coords_2 = board.columns[x+dx+dx] + board.rows[y+dy+dy]
			
			if board[coords_1] != -self.color:
				# If the neighbour square is not of the opposite color
				return False
			
			# We don't need to make this test as it will be tested in the next
			# recursive iteration if needed
			
#			elif board[coords_2] == 0:
#				# If the following square is empty, the line is not valid
#				return False
				
			elif board[coords_2] == self.color:
				# If the following square is of the right colour, then the line
				# is matching, and the move is valid
				return True
			else:
				# Otherwise, we need to test one step further
				return valid_line(board, x+dx, y+dy, dx, dy)
				
		
		valid_moves = []
		for i in range(len(cols)):
			for j in range(len(rows)):
				coords = cols[i] + rows[j]
				
				if board[coords] != 0:
					# If there is already a piece on the square, the move is 
					# not valid
					continue
				else:
					nw = valid_line(board, i, j, -1, -1)
					nn = valid_line(board, i, j,  0, -1)
					ne = valid_line(board, i, j,  1, -1)
					
					ww = valid_line(board, i, j, -1,  0)
					ee = valid_line(board, i, j,  1,  0)
					
					sw = valid_line(board, i, j, -1,  1)
					ss = valid_line(board, i, j,  0,  1)
					se = valid_line(board, i, j,  1,  1)
				
				if nw or nn or ne or ww or ee or sw or ss or se:
					valid_moves.append(coords)
					
		return valid_moves
		
class Human(Player):
	
	def __init__(self):
		super().__init__(True)
		
	def play(self, board):
		possible_moves = self._possible_moves(board)
		
		if not possible_moves:
			# If the list is empty, then there is no possible move, and the
			# game is finished
			return None
		
		move = None
		while move not in possible_moves:
			print(board)
			print('\n{}, what move do you want to play ?'.format(self.name))
			move = input('> ')
			print('\n')
			
			if move not in possible_moves:
				print('#### WARNING - Invalid move ####\n')
			
		return move

class Bot(Player):
	
	def __init__(self):
		super().__init__(False)
	
class RandomBot(Bot):
	
	def __init__(self):
		super().__init__()
		
	def play(self, board):
		possible_moves = self._possible_moves(board)
		
		if not possible_moves:
			return None
		else:
			print(board)
			return random.choice(possible_moves)
		

class Board:
	
	length = 8
	width = 8
	columns = 'ABCDEFGH'
	rows = '12345678'
	
	def __init__(self):		
		self._board = self._empty_board()
		
	@staticmethod
	def _empty_board():
		return [[0 for i in range(Board.width)] for j in range(Board.length)]
	
	def init_position(self):
		"""
		Initialize the board with the standard Othello position
		"""
		
		# Starting position, 1 is for WHITE, -1 is for BLACK
		self['D4'] = self['E5'] = 1
		self['D5'] = self['E4'] = -1
		
	def _is_valid_key(self, key):
		"""
		Check if the given key is valid, with the format XY. X is the
		column, between A and H, and Y is the row, between 1 and 8.
		"""
		
		# If the key is not a string
		if not isinstance(key, str):
			return False
		else:
			key = str.upper(key)
		
		# If the given key does not match the standard notation XY
		if len(key) != 2:
			return False
		
		# If the key is out of the board
		if key[0] not in self.columns or key[1] not in self.rows:
			return False
		
		# Otherwise the key is valid
		return True
	
	def _index_from_key(self, key):
		"""
		Return the board coordinates i, j corresponding to the key XY
		"""
		
		return self.columns.index(str.upper(key[0])), self.rows.index(key[1])
		
	def __setitem__(self, key, value):
		if not self._is_valid_key(key):
			raise KeyError('The key is not valid')
			
		if not value in [-1, 0, 1]:
			raise ValueError('The value must be -1, 0 or 1')
			
		x, y = self._index_from_key(key)
		self._board[x][y] = value

	def __getitem__(self, key):
		if not self._is_valid_key(key):
			raise KeyError
		
		x, y = self._index_from_key(key)
		return self._board[x][y]
	
	def __str__(self):
		"""
		Convert the table into a readable board, with coordinates
		"""
		
		def mapping(x):
			if x == 1:
				# WHITE
				return 'O'
			elif x == -1:
				# BLACK
				return 'X'
			else:
				# Empty
				return '-'
		
		s = 'BLACK - X\n'
		s += 'WHITE - O\n\n'
		for j in self.rows:
			s += j
			s += ' '
			s += ''.join(mapping(self[i+j]) for i in self.columns)
			s += '\n'
		s += '\n  '
		return s + self.columns
			

class Game:
	
	def __init__(self, player1, player2, max_time=30):
		self.white = player1
		self.white.color = 1
		
		self.black = player2
		self.black.color = -1
		
		self.max_time = 30
	
		self.board = Board()
		self.board.init_position()
		
		self.white_turn = False # True is for WHITE's turn, False is for BLACK's
		
	def start(self):
		while self.play_turn():
			continue
		
		self.finish()
		
	def play_turn(self):
		if self.white_turn:
			move = self.white.play(self.board)
			symbol = 1
		else:
			move = self.black.play(self.board)
			symbol = -1
			
		if move:
			self.board[move] = symbol
			self.update_board(move)
			
			self.white_turn = not self.white_turn
			
			return True
		else:
			return False
			
	def update_board(self, move):
		"""
		Change pieces color for the given move
		"""
		pass
	
	def finish(self):
		print('#### Game is finished ####\n')
		
		print(self.board)
		
		white_count = sum(sum(self.board[i+j] == 1 for i in self.board.columns)
					      for j in self.board.rows)
		black_count = sum(sum(self.board[i+j] == -1 for i in self.board.columns)
						  for j in self.board.rows)
		
		print('\n')
		print('WHITE {} - {} BLACK'.format(white_count, black_count))
		print('\n')
		
		if white_count == black_count:
			print('This is a DRAW\n')
		elif white_count > black_count:
			print('WHITE wins')
		else:
			print('BLACK wins')
		
			
			
			