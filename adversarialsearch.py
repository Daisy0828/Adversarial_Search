from adversarialsearchproblem import AdversarialSearchProblem
from gamedag import GameDAG, DAGState
from sys import maxsize


def minimax(asp):
    """
    Implement the minimax algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))

    """

    ret = None
    value = - maxsize
    state = asp.get_start_state()
    #print("state is", state)
    ptm = state.player_to_move()
    #print("player to move is", ptm)

    for action in asp.get_available_actions(state):
        temp  = min_value(asp, asp.transition(state, action), ptm)
        if temp > value:
            value = temp
            ret = action
    return ret

def min_value(asp, state, ptm):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[ptm]
    value = maxsize
    for action in asp.get_available_actions(state):
        value = min(max_value(asp, asp.transition(state, action), ptm), value)
    return value

def max_value(asp, state, ptm):
    if asp.is_terminal_state(state):
        return asp.evaluate_state(state)[ptm]
    value = - maxsize
    for action in asp.get_available_actions(state):
        value = max(min_value(asp, asp.transition(state, action), ptm), value)
    return value

def alpha_beta(asp):
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    state = asp.get_start_state()
    ptm = state.player_to_move()
    alpha = -maxsize
    beta = maxsize
    action, value = max_value_ab(asp, state, alpha, beta, ptm)
    return action


def min_value_ab(asp, state, alpha, beta, ptm):
    if asp.is_terminal_state(state):
        return None, asp.evaluate_state(state)[ptm]

    value = maxsize
    ret = None
    for action in asp.get_available_actions(state):
        temp_action, temp_value = max_value_ab(asp, asp.transition(state, action), alpha, beta, ptm)
        if temp_value < value:
            value = temp_value
            ret = action
        if value <= alpha:
            return ret, value
        beta = min(beta, value)
    return ret, value


def max_value_ab(asp, state, alpha, beta, ptm):
    if asp.is_terminal_state(state):
        return None, asp.evaluate_state(state)[ptm]

    value = - maxsize
    ret = None
    for action in asp.get_available_actions(state):
        temp_action, temp_value = min_value_ab(asp, asp.transition(state, action), alpha, beta, ptm)
        if temp_value > value:
            value = temp_value
            ret = action
        if value >= beta:
            return ret, value
        alpha = max(alpha, value)
    return ret, value


def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Inputs:
        asp - an AdversarialSearchProblem
        cutoff_ply- an Integer that determines when to cutoff the search
            and use eval_func.
            For example, when cutoff_ply = 1, use eval_func to evaluate
            states that result from your first move. When cutoff_ply = 2, use
            eval_func to evaluate states that result from your opponent's
            first move. When cutoff_ply = 3 use eval_func to evaluate the
            states that result from your second move.
            You may assume that cutoff_ply > 0.
        eval_func - a function that takes in a GameState and outputs
            a real number indicating how good that state is for the
            player who is using alpha_beta_cutoff to choose their action.
            You do not need to implement this function, as it should be provided by
            whomever is calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The eval_func we provide
            does not handle terminal states, so evaluate terminal states the
            same way you evaluated them in the previous algorithms.

    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    state = asp.get_start_state()
    ptm = state.player_to_move()
    alpha = -maxsize
    beta = maxsize
    action, value = max_value_abcut(asp, state, alpha, beta, ptm, cutoff_ply, eval_func)
    return action



def min_value_abcut(asp, state, alpha, beta, ptm, cutoff_ply, eval_func):
    if  asp.is_terminal_state(state):
        return None, asp.evaluate_state(state)[ptm]
    if cutoff_ply == 0:
        return None, eval_func(state)

    value = maxsize
    ret = None
    for action in asp.get_available_actions(state):
        temp_action, temp_value = max_value_abcut(asp, asp.transition(state, action), alpha, beta, ptm, cutoff_ply - 1, eval_func)
        if temp_value < value:
            value = temp_value
            ret = action
        if value <= alpha:
            return ret, value
        beta = min(beta, value)
    return ret, value


def max_value_abcut(asp, state, alpha, beta, ptm, cutoff_ply, eval_func):
    if  asp.is_terminal_state(state):
        return None, asp.evaluate_state(state)[ptm]
    if cutoff_ply == 0:
        return None, eval_func(state)

    value = - maxsize
    ret = None
    for action in asp.get_available_actions(state):
        temp_action, temp_value = min_value_abcut(asp, asp.transition(state, action), alpha, beta, ptm, cutoff_ply - 1, eval_func)
        if temp_value > value:
            value = temp_value
            ret = action
        if value >= beta:
            return ret, value
        alpha = max(alpha, value)
    return ret, value


def general_minimax(asp):
    """
    Implement the generalization of the minimax algorithm that was
    discussed in the handout, making no assumptions about the
    number of players or reward structure of the given game.

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    state = asp.get_start_state()
    ptm = state.player_to_move()

    ret, value = general_minimax_recursion(asp, state, ptm)
    return ret


def general_minimax_recursion(asp, state, ptm):
    if asp.is_terminal_state(state):
        return None, asp.evaluate_state(state)[ptm]
    if state.player_to_move() == ptm:
        value = -maxsize
        ret = None
        for action in asp.get_available_actions(state):
            temp_action, temp_value = general_minimax_recursion(asp, asp.transition(state, action), ptm)
            if temp_value > value:
                value = temp_value
                ret = action
        return ret, value
    else:
        value = maxsize
        ret = None
        for action in asp.get_available_actions(state):
            temp_action, temp_value = general_minimax_recursion(asp, asp.transition(state, action), ptm)
            if temp_value < value:
                value = temp_value
                ret = action
        return ret, value



