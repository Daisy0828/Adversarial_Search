import unittest
from adversarialsearchproblem import AdversarialSearchProblem
from gamedag import GameDAG, DAGState
from adversarialsearch import minimax, alpha_beta, alpha_beta_cutoff, general_minimax


class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take
    """

    def _get_test_dag(self):
        """
    	An example of an implemented GameDAG from the gamedag class.
    	Look at handout in section 3.3 to see visualization of the tree.

        Output: GameDAG to be used for testing
    	"""
        matrix = [
            [False, True, True, False, False, False, False],
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, True, True],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        ]
        start_state = DAGState(0, 0)
        terminal_indices = set([3, 4, 5, 6])
        evaluations_at_terminal = {3: [-1, 1], 4: [-2, 2], 5: [-3, 3], 6: [-4, 4]}
        turns = [0, 1, 1, 0, 0, 0, 0]
        dag = GameDAG(
            matrix, start_state, terminal_indices, evaluations_at_terminal, turns
        )
        return dag

    def _check_result(self, result, dag):
        """
            Tests whether the result is one of the possible actions
            of the dag.
            Input:
                result- the return value of an adversarial search problem.
                    This should be an action
                dag- the GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potentialActions = dag.get_available_actions(start_state)
        self.assertTrue(
            result in potentialActions, "Output should be an available action"
        )

    def _general_check_algorithm(self, algorithm):
        """
            instantiates an adversarial search problem (DAG), and
            checks that the result is an action
            Input:
                algorithm- a function that takes in an asp and returns an
                action
        """
        dag = self._get_test_dag()
        result = algorithm(dag)
        self._check_result(result, dag)

    def _dummy_eval_func(self, gameState):
        return 0

    def test_minimax(self):
        self._general_check_algorithm(minimax)
        print("minimax passes basic I/O specifications")

    def test_alpha_beta(self):
        self._general_check_algorithm(alpha_beta)
        print("alpha-beta passes basic I/O specifications")

    def test_alpha_beta_cutoff(self):
        dag = self._get_test_dag()
        cutoff = 1
        result = alpha_beta_cutoff(dag, cutoff, self._dummy_eval_func)
        self._check_result(result, dag)
        print("alpha-beta cutoff passes basic I/O specifications")

    def test_general_minimax(self):
        self._general_check_algorithm(general_minimax)
        print("general minimax passes basic I/O specifications")


class CorrectActionTest(unittest.TestCase):
    """
    Tests "correct" action to take for adversarial search implementations.
    Contains simple test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns the correct action for this simple DAG.
    """

    def _get_test_dag_2(self):
        """
    	An example of an implemented GameDAG from the gamedag class.
    	Look at handout in section 3.3 to see visualization of the tree.

        Output: GameDAG to be used for testing
    	"""
        matrix = [
            [False, True, True, True,False,False, False, False, False, False, False, False, False,],
            [False, False, False, False, True, True, True, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, True, True,True,False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, True,True,True,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
            [False, False, False, False, False, False, False, False, False, False, False, False, False,],
        ]
        start_state = DAGState(0, 0)
        terminal_indices = set([4, 5, 6, 7, 8, 9, 10, 11, 12])
        constant_evaluations_at_terminal = {
        4: [-1, 1],
        5: [-4, 4],
        6: [-5, 5],
        7: [2, -2],
        8: [3, -3],
        9: [8, -8],
        10: [-16, 16],
        11: [-3, 3],
        12: [-16, 16],
        }
        variable_evaluations_at_terminal = {
            4: [-1, 1],
            5: [-3, 3],
            6: [-5, 5],
            7: [12, -12],
            8: [11, -13],
            9: [3, -18],
            10: [20, 16],
            11: [-3, 3],
            12: [-16, 16],
        }

        turns = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        constant_dag = GameDAG( matrix, start_state, terminal_indices, constant_evaluations_at_terminal, turns)
        variable_dag = GameDAG( matrix, start_state, terminal_indices, variable_evaluations_at_terminal, turns)
        return constant_dag, variable_dag

    def _output_check_result(self, result, dag):
        """
            Tests whether the result is the "correct" action
            of the dag.
            Input:
                result- the action of an adversarial search problem.
                dag- the GameDAG that was used to test the algorithm
        """
        self.assertIsNotNone(result, "Output should not be None")
        solution = 2
        self.assertTrue(
            result == solution, "Output producing incorrect action"
        )
    def _output_check_algorithm(self, algorithm, constant):
        """
            instantiates an adversarial search problem (DAG), and
            checks that the result is the "correct" action for a constant
            Input:
                algorithm- a function that takes in an asp and returns an
                action
        """
        constant_dag, variable_dag = self._get_test_dag_2()
        if constant:
            result = algorithm(constant_dag)
            self._output_check_result(result, constant_dag)
        else:
            result = algorithm(variable_dag)
            self._output_check_result(result, variable_dag)

    def _dummy_eval_func(self, gameState):
        return 0

    def test_minimax(self):
        self._output_check_algorithm(minimax, True)
        print("minimax produces correct action for simple DAG")

    def test_alpha_beta(self):
        self._output_check_algorithm(alpha_beta, True)
        print("alpha-beta produces correct action for simple DAG")

    def test_alpha_beta_cutoff(self):
        constant_dag, variable_dag = self._get_test_dag_2()
        cutoff = 2
        result = alpha_beta_cutoff(constant_dag, cutoff, self._dummy_eval_func)
        self._output_check_result(result, constant_dag)
        print("alpha-beta cutoff produces correct action for simple DAG")



    def test_general_minimax(self):
        self._output_check_algorithm(general_minimax, False)
        print("general minimax produces correct action for simple DAG")

if __name__ == "__main__":
    unittest.main()
