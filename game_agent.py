"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # TODO: finish this function!
    if game.is_winner(player): # check to see if player is in state winner
        #print("You win!")
        return math.inf # abstraction of score, +inf equates to a win
    elif game.is_loser(player):
        #print("You lose!")
        return -math.inf # abstraction of score, -inf equates to a loss

    # Opponent
    opponent = game.get_opponent(player)

    # Remaining spaces left on the board
    rem_spaces = len(game.get_blank_spaces())

    # number of agent's available moves
    no_moves = len(game.get_legal_moves(player))

    # number of opponent's available moves
    opp_moves = len(game.get_legal_moves(opponent))

    # evaluation of board "goodness"
    # using moves available to both players
    # Idea is player chooses moves with scores that maximise whilst minimizing
    # evaluate board states and positions as scores
    board_score = no_moves - opp_moves
    score = board_score/rem_spaces

    return float(score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # TODO: finish this function!
    if game.is_loser(player):
        #print("You lose!")
        return -math.inf
    if game.is_winner(player):
        #print("You win")
        return math.inf

    # center
    width = game.width / 2
    height = game.height / 2

    # Opponent
    opponent = game.get_opponent(player)
    opp_y_coord, opp_x_coord = game.get_player_location(opponent)
    opp_x_eval = (width - float(opp_x_coord)) ** 2
    opp_y_eval = (height - float(opp_y_coord)) ** 2
    opp_center_eval = float(opp_x_eval + opp_y_eval)

    # Remaining spaces left on the board
    rem_spaces = len(game.get_blank_spaces())

    # number of agent's available moves
    no_moves = len(game.get_legal_moves(player))

    # number of opponent's available moves
    opp_moves = len(game.get_legal_moves(opponent))

    # evaluation of board "goodness"
    # using moves available to both players
    # Idea is player chooses moves with scores that maximise whilst minimizing
    # evaluate board states and positions as scores
    opp_score = opp_moves * 2 - opp_center_eval
    score = no_moves - opp_score/rem_spaces
    return float(score)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # TODO: finish this function!
    if game.is_winner(player):  # check to see if player is in state winner
        #print("You win!")
        return math.inf  # abstraction of score, +inf equates to a win
    elif game.is_loser(player):
        #print("You lose!")
        return -math.inf  # abstraction of score, -inf equates to a loss

    # center
    width = game.width / 2
    height = game.height / 2

    # Opponent
    opponent = game.get_opponent(player)
    opp_y_coord, opp_x_coord = game.get_player_location(opponent)
    opp_x_eval = (width - float(opp_x_coord)) ** 2
    opp_y_eval = (height - float(opp_y_coord)) ** 2
    opp_center_eval = float(opp_x_eval + opp_y_eval)

    # Remaining spaces left on the board
    rem_spaces = len(game.get_blank_spaces())

    # number of agent's available moves
    no_moves = len(game.get_legal_moves(player))

    # number of opponent's available moves
    opp_moves = len(game.get_legal_moves(opponent))

    # evaluation of board "goodness"
    # using moves available to both players
    # Idea is player chooses moves with scores that maximise whilst minimizing
    # evaluate board states and positions as scores
    score = no_moves * 2 - opp_center_eval
    return float(score)


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
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # TODO: finish this function!
        # raise NotImplementedError

        legal_moves = game.get_legal_moves()  # obtain legal moves available to the board
        best_move = (-1,-1)  # initialisation of best move
        best_score = -math.inf  # abstraction of infinity

        for m in legal_moves:  # for each ACTION, create a new state for its outcome, RESULT
            new_state = game.forecast_move(m)
            score = self.min_value(new_state, depth - 1)  # recursion to calculate the score of that state
            if score > best_score:
                best_move = m
                best_score = score
        return best_move

    def min_value(self, game, depth):
        """
        Helper function for calculating the minimising score
        """
        if self.time_left() < self.TIMER_THRESHOLD:  # Timeout check
            raise SearchTimeout()

        if game.is_loser(self) or game.is_winner(self) or depth == 0:  # Terminal test, checks base cases
            return self.score(game,self)  # returns the score, UTILITY of the current state
        legal_moves = game.get_legal_moves()  # obtain all legal moves for game, ACTIONs that can be taken
        best_score = math.inf  # abstraction assignment of infinity(highest possible value for MIN score)
        for m in legal_moves:  # iterate through all available actions
            new_state = game.forecast_move(m)  # for each available move, forecast the resulting state from that ACTION
            # RESULT of ACTION
            score = self.max_value(new_state, depth - 1)  # recursively uses the new state
            best_score = min(best_score,score)  # calculates the minimizing score between the states
        return best_score  # propagates minimizing score for given state

    def max_value(self, game, depth):
        """
        Helper function for calculating the maximising score
        """
        if self.time_left() < self.TIMER_THRESHOLD:   # Timeout check
            raise SearchTimeout()

        if game.is_loser(self) or game.is_winner(self) or depth == 0:  # Terminal test, checks base cases
            return self.score(game,self)  # returns the score, UTILITY of the current state
        legal_moves = game.get_legal_moves()  # obtain all legal moves for game, ACTIONs that can be taken
        best_score = -math.inf  # abstraction assignment of neg. infinity(lowest possible value for MAX score)
        for m in legal_moves:  # iterate through all available actions
            new_state = game.forecast_move(m)  # for each available move, forecast the resulting state from that ACTION
            # RESULT of ACTION
            score = self.max_value(new_state, depth - 1)  # recursively uses the new state
            best_score = max(best_score,score)  # calculates the minimizing score between the states
        return best_score  # propagates minimizing score for given state


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

        # TODO: finish this function!
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        # The try/except block will automatically catch the exception# raised when the timer is about to expire.
        # Implementation of Iterative Deepening Search

        try:
            depth = 0  # initialisation of depth
            while True:
                # runs along as game is still active
                depth += 1  # increment depth after each search
                best_move = self.alphabeta(game, depth)  # apply alpha beta to search

        except SearchTimeout:  # cutoff - when timer runs out
            pass  # Handle any actions required after timeout as needed
            # Failure

        # Return the best move from the last completed search iteration
        return best_move  # Solution

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        # raise NotImplementedError
        legal_moves = game.get_legal_moves()  # obtain list of all available moves on the board
        best_move = (-1, -1)  # initialise best move in case of error
        best_score = -math.inf  # abstraction assignment of infinity

        for m in legal_moves:  # iterate through all moves available to the board - ACTIONS
            new_state = game.forecast_move(m)  # capture new game state - RESULT for each move - ACTION
            score = self.min_value(new_state, depth - 1, alpha, beta)  # recursive call to min_value to test new state along with alpha and beta
            if score > best_score:  # check to see if score of move is better than current best score
                best_move = m  # update best move to the current move
                best_score = score  # update best score to current state's score
            if best_score >= beta:  # check best move against lower bound
                return m  # return move
            alpha = max(alpha, best_score)  # calculate max between alpha and best score, work out higher bound for current state
        return best_move  # return best move for player

    def max_value(self, game, depth, alpha, beta):
        """
        Helper function for calculating the upper boundary, beta used for pruning
        """
        if self.time_left() < self.TIMER_THRESHOLD:  # Timeout check
            raise SearchTimeout()

        if game.is_loser(self) or game.is_winner(self) or depth == 0:  # Terminal test, checks base cases
            return self.score(game,self)  # returns the score, UTILITY of the current state

        legal_moves = game.get_legal_moves()  # obtain all the available moves on the board
        best_score = -math.inf  # abstraction assignment of neg. infinity

        for m in legal_moves:  # iterate through available moves - ACTIONS available to the state
            new_state = game.forecast_move(m)
            # for each move - ACTION, create the outcome of that move - RESULT of each ACTION resulting in a new state
            score = self.min_value(new_state, depth - 1, alpha, beta)  # recursive call to min - using new state, alpha and beta
            best_score = max(best_score, score)  # calculate max between best_score and score
            if best_score >= beta:  # check if best score is greater than or equal to beta
                return best_score  # return best score
            alpha = max(alpha, best_score)  # calculate max between alpha and best_score
        return best_score  # propagate max and return its value

    def min_value(self, game, depth, alpha, beta):
        """
        Helper function for calculating the lower boundary, alpha used for pruning
        """
        if self.time_left() < self.TIMER_THRESHOLD:  # Timeout check
            raise SearchTimeout()

        if game.is_loser(self) or game.is_winner(self) or depth == 0:  # Terminal test, checks base cases
            return self.score(game,self)  # returns the score, UTILITY of the current state

        legal_moves = game.get_legal_moves()  # obtain all the available moves on the board
        best_score = math.inf  # abstraction assignment of infinity

        for m in legal_moves:  # iterate through available moves - ACTIONS available to the state
            new_state = game.forecast_move(m)
            # for each move - ACTION, create the outcome of that move - RESULT of each ACTION resulting in a new state
            score = self.max_value(new_state, depth - 1, alpha, beta)  # recursive call to max - using new state, alpha and beta
            best_score = min(best_score, score)  # calculate min between best_score and score
            if best_score <= alpha:  # check if best score is less than or equal to alpha
                return best_score  # return best score
            beta = min(beta, best_score)  # calculate min between alpha and best_score
        return best_score  # propagate min and return its value
