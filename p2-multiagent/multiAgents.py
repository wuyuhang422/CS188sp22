# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves)
        # print(scores)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        def dis(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        for x in newGhostStates:
            if dis(newPos, x.getPosition()) <= 1:
                return -float("inf")
        mn = -float("inf") if len(newFood) == 0 else min([dis(newPos, x) for x in newFood])
        return successorGameState.getScore()-mn/10

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        def minimax_search(gameState, dep, now):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore(), None
            if dep > self.depth:
                return self.evaluationFunction(gameState), None
            comb = max if now == 0 else min
            legal_moves = gameState.getLegalActions(now)
            scores = [minimax_search(gameState.generateSuccessor(now, act), dep+(now == num_agents-1), (now+1)%num_agents)[0] for act in legal_moves]
            bestScore = comb(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            return bestScore, legal_moves[chosenIndex]
        return minimax_search(gameState, 1, 0)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        def minimax_search_with_pruning(gameState, dep, now, alpha=-float("inf"), beta=float("inf")):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore(), None
            if dep > self.depth:
                return self.evaluationFunction(gameState), None
            legal_moves = gameState.getLegalActions(now)
            bestMove = None
            if now == 0: # max node
                bestScore = -float("inf")
                for act in legal_moves:
                    tmp = minimax_search_with_pruning(gameState.generateSuccessor(now, act), dep+(now == num_agents-1), (now+1)%num_agents, alpha, beta)[0]
                    if bestScore < tmp:
                        bestScore = tmp
                        bestMove = act
                    if bestScore > beta:
                        break
                    alpha = max(alpha, bestScore)
            else: # min node
                bestScore = float("inf")
                for act in legal_moves:
                    tmp = minimax_search_with_pruning(gameState.generateSuccessor(now, act), dep+(now == num_agents-1), (now+1)%num_agents, alpha, beta)[0]
                    if bestScore > tmp:
                        bestScore = tmp
                        bestMove = act
                    if bestScore < alpha:
                        break
                    beta = min(beta, bestScore)
            return bestScore, bestMove
        return minimax_search_with_pruning(gameState, 1, 0)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        def minimax_search(gameState, dep, now):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore(), None
            if dep > self.depth:
                return self.evaluationFunction(gameState), None
            legal_moves = gameState.getLegalActions(now)
            scores = [minimax_search(gameState.generateSuccessor(now, act), dep+(now == num_agents-1), (now+1)%num_agents)[0] for act in legal_moves]
            if now == 0:
                bestScore = max(scores)
                bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
                chosenIndex = random.choice(bestIndices)
                return bestScore, legal_moves[chosenIndex]
            else:
                expectedScore = sum(scores)/len(legal_moves)
                return expectedScore, None
        return minimax_search(gameState, 1, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    def dis(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    mnGhost = float("inf")
    for x in GhostStates:
        tmp = dis(position, x.getPosition())
        mnGhost = min(mnGhost, tmp)
        if tmp <= 1:
            return -float("inf") if x.scaredTimer <= 1 else float("inf")
    mnfood = -float("inf") if len(food) == 0 else min([dis(position, x) for x in food])
    return currentGameState.getScore()-mnfood/10+mnGhost/10
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
