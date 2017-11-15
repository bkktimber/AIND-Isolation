import random
from math import log


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Custom Hueristic 1 - Moves Difference

    This Custom Hueristic based on goal toward the end game state which
    opponent runs out of legal move.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    # get avaliable moves for each player
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    # return different between # of my agent's move and oppenent's
    return float(own_moves - opp_moves)


def custom_score_2(game, player):
    """Custom Hueristic 2 - Avaliable Move Ratio
    This hueristic prefer state that diffrence between avaliable moves of 
    2 players is greatest.

    The end game state of choice is that opponent's move (opp_moves) is 0
    ratio of own_moves to opp_moves is inf.
    On the other hand, ratio will be 0 aka. we lose.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    # get avaliable moves for each player
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    # shortcut to definite state:
    # 1. my agent win -> return very high score
    if opp_moves == 0:
        return float("inf")
    # 2. opponenent's agent win -> return very low score
    elif own_moves == 0:
        return float("-inf")

    # score: avaliable moves ratio
    return float(own_moves/opp_moves)


def custom_score_3(game, player):
    """Custom Hueristic 3 - Natural Log of Avaliable Move Ratio
    This hueristic is modified version of Avaliable Move Ratio.

    Since we know that log function range is (0, inf) and grows exponetially
    This hueristic returns extreamly high number when game moves toward
    my agent's victory. 

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # get avaliable moves for each player
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    # shortcut to definite state:
    # 1. my agent win -> return very high score
    if opp_moves == 0:
        return float("inf")
    # 2. opponenent's agent win -> return very low score
    elif own_moves == 0:
        return float("-inf")

    # score: log of avaliable moves ratio
    return float(log(own_moves/opp_moves))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    def minimax(self, game, depth):
        
        # Timeout Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Helper Function1: max
        def max_value(self, game, depth):
            """This is helper function for minimax
            Max_value (self, game, depth)

            Parameters:
            game: game state
            depth: search depth

            Find minimum score of each game state corresponding to its legal moves
            Return score of that state when search complete.
            """
            
            # Timeout Check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Get legal moves
            valid_moves = game.get_legal_moves()            
            # Best possible score -> initiated at -inf, the lowest score possible
            best_value = float("-inf")

            # Terminal State:
            # When search reaches search limit or no legal moves left
            # Return score of terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)
            
            # Search each move in legal moves
            for move in valid_moves:

                # Update best possible value with current best or search value 
                best_value = max(best_value, min_value(self, game.forecast_move(move), depth-1))

            # Return best value (in this case max value)
            return best_value
        
        def min_value(self, game, depth):
            """This is helper function for minimax
                Min_value (self, game, depth)

                Parameters:
                game: game state
                depth: search depth

                Find minimum score of each game state corresponding to its legal moves
                Return score of that state when search complete.
            """

            # Timeout Check 
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Get legal moves
            valid_moves = game.get_legal_moves()
            # Best possible score -> initiated at inf, the highest score possible
            best_value = float("inf")

            # Terminal State:
            # When search reaches search limit or no legal moves left
            # Return score of terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)

            # Search each move in legal moves
            for move in valid_moves:
                
                # Update best possible value with current best or search value 
                best_value = min(best_value, max_value(self, game.forecast_move(move), depth-1))
            
            # Return best value (in this case min value) 
            return best_value

        # Main MiniMax Function
        # Get legal moves
        valid_moves = game.get_legal_moves()

        # Best possible move -> initiated at (-1,-1)
        # Best possible score -> initiated at -inf, the lowest score possible
        best_score = float("-inf")
        best_move = (-1, -1)

        # Terminal State:
        # When no legal moves left return (-1, -1) move to forfeit
        if (not valid_moves):
            return (-1, -1)

        # Search best move from each move in legal moves
        # Using minimax by first call min_value (helper function)
        # While searching, if any move return better score than current best
        # score, set that move and corresponding score as new target.
        for move in valid_moves:
            score = min_value(self, game.forecast_move(move), depth-1)
            if score > best_score:
                best_move = move
                best_score = score

        # At the end of search, return best move for that state.
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # Opening Book
        # Check if my agent is first to move
        # If yes, use opening book
        if (game._board_state[-1] == None):
            if (not game.get_legal_moves()):
                return best_move
            else:
                best_move = (4,4)
                return best_move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            # Iterative Deepning, stop when timeout
            depth = 0
            while (True):
                depth += 1
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_Player = True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def max_value(self, game, depth, alpha, beta):
            """This is helper function for alpha-beta prunnig on minimax
                Min_value (self, game, depth, alpha, beta)

                Parameters:
                game: game state
                depth: search depth
                alpha: search upper limit
                beta: search lower limit

                Find maximum score of each game state corresponding to its legal moves
                Set new alpha (search upper limit) if find score higher than current limit
                Return score of that state when search complete.
            """
            
            # Timeout Check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # Get legal moves
            valid_moves = game.get_legal_moves()
            # Best possible score -> initiated at inf, the highest score possible
            best_value = float("-inf")
            
            # Terminal State:
            # When search reaches search limit or no legal moves left
            # Return score of terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)
            
            # Search each move in legal moves
            for move in valid_moves:

                # Update best possible value with current best or search value 
                best_value = max(best_value, min_value(self, game.forecast_move(move), depth-1, alpha, beta))
                
                # Update beta when best bossible value is equal or higher than beta
                if (best_value >= beta):
                    return best_value

                # Update alpha if best possible value is higher than alpha
                alpha = max(best_value, alpha)
                
            # Return best value (in this case max value) 
            return best_value
        
        def min_value(self, game, depth, alpha, beta):
            """This is helper function for alpha-beta prunnig on minimax
                Min_value (self, game, depth, alpha, beta)

                Parameters:
                game: game state
                depth: search depth
                alpha: search upper limit
                beta: search lower limit

                Find minimum score of each game state corresponding to its legal moves
                Set new beta (search lower limit) if find score lower than current limit
                Return score of that state when search complete.
            """
            
            # Timeout Check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            # Get legal moves
            valid_moves = game.get_legal_moves()
            # Best possible score -> initiated at inf, the highest score possible
            best_value = float("inf")
            
            # Terminal State:
            # When search reaches search limit or no legal moves left
            # Return score of terminal state
            if (depth == 0) or (not valid_moves):
                return self.score(game, self)
            
            # Search each move in legal moves
            for move in valid_moves:
                
                # Update best possible value with current best or search value 
                best_value = min(best_value, max_value(self, game.forecast_move(move), depth-1, alpha, beta))
                
                # Update beta when best bossible value is equal or lower than alpha
                if (best_value <= alpha):
                    return best_value
                
                # Update alpha if best possible value is lower than beta
                beta = min(best_value, beta)
                
            return best_value
        
        # Timeout Check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Main MiniMax Function
        # Get legal moves
        valid_moves = game.get_legal_moves()

        # Best possible move -> initiated at (-1,-1)
        # Best possible score -> initiated at -inf, the lowest score possible
        best_score = float("-inf")
        best_move = (-1, -1)

        # Terminal State:
        # When no legal moves left return (-1, -1) move to forfeit
        if (depth == 0) or (not valid_moves):
            return (-1, -1)
        
        # Search best move from each move in legal moves
        # Using minimax by first call min_value (helper function)
        # While searching, if any move return better score than current best
        # core, set that move and corresponding score as new target also set new upper limit with maximum score so far
        # Search ends when score is higher than beta
        for move in valid_moves:
            score = min_value(self, game.forecast_move(move), depth -1, alpha, beta)
            if (score > best_score):
                best_score = score
                best_move = move
                alpha = max(alpha, score)
                if best_score >= beta:
                    return best_move

        return best_move
