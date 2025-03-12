# Parker Hix and Joy Mosisa

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

import random, util
import sys # We use sys only to use a very small value to prevent division by zero (float_info.epsilon)

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

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluates a game state during the Expectiminimax algorithm.
        Returns a numerical score where:
        - Higher scores encourage Pac-Man to pursue that state
        - Lower scores encourage Pac-Man to avoid that state
        """

        # Extract key information about the game state after taking the action
        successorGameState = currentGameState.generatePacmanSuccessor(action)  # The new state after the move
        newPosition = successorGameState.getPacmanPosition()                  # Pac-Man's new coordinates
        foodPellet = successorGameState.getFood()                             # Grid of remaining food
        ghostStates = successorGameState.getGhostStates()                     # List of ghost states

        # Start with the base score from the successor state
        evaluation = successorGameState.getScore()

        # Evaluate ghost proximity
        for ghostState in ghostStates:
            ghostPosition = ghostState.getPosition()                          # Current position of this ghost
            ghostDistance = util.manhattanDistance(newPosition, ghostPosition) # Distance from Pac-Man to ghost

            # Penalize being too close to ghosts
            if ghostDistance <= 1:
                evaluation -= 100  # Apply a large penalty for nearby ghosts (within 1 grid space)

        # Evaluate food proximity and count
        remainingFood = foodPellet.asList()  # Convert food grid to a list of food positions
        if remainingFood:                    # Check if any food pellets remain
            # Find the distance to the nearest food pellet
            closestFoodDist = min([util.manhattanDistance(newPosition, food) for food in remainingFood])
            # Boost score based on proximity to food
            # Closer food gives a larger bonus; distant food contributes almost nothing
            closestFoodDist = max(closestFoodDist, sys.float_info.epsilon)  # Prevent division by zero with a tiny value
            evaluation += 1.0 / closestFoodDist

        # Penalize based on total remaining food
        evaluation -= len(remainingFood)  # Subtract 1 point per uneaten food pellet

        # Return the final evaluation score
        return evaluation

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
        #Pseudocode for minimax algorithm
        #def value(state):
        # if state is terminal: return utility(state)
        # if next agent is max: return max-value(state)
        # if next agent is min: return min-value(state)

        #def max-value(state):
        # v = -inf
        # for each successor of state:
        # v = max(v, value(successor))
        # return v

        #def min-value(state):
        # v = inf
        # for each successor of state:
        # v = min(v, value(successor))
        # return v

        #Return the value of the state
        def value(state, depth, agentIndex):
            #If state is terminal or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                #Return the evaluation function of the state
                return self.evaluationFunction(state)
            #If Pacman's turn (maximizing agent)
            if agentIndex == 0:
                #Return the max value of the state
                return maxValue(state, depth)
            #If Ghosts' turn (minimizing agents)
            else:
                #Return the min value of the state
                return minValue(state, depth, agentIndex)

        #Return the max value of the state
        def maxValue(state, depth):
            #Initial value
            v = float('-inf')
            #Loop through legal actions of Pacman
            for action in state.getLegalActions(0):
                #Calculate max value
                v = max(v, value(state.generateSuccessor(0, action), depth, 1))
            #Return the max value
            return v

        #Return the min value of the state
        def minValue(state, depth, agentIndex):
            #Initial value
            v = float('inf')
            #Get next agent
            nextAgent = agentIndex + 1
            #If next agent is the last agent
            if nextAgent == state.getNumAgents():
                #Reset next agent to 0
                nextAgent = 0
                #Increase depth
                depth += 1
            #Loop through legal actions of the agent
            for action in state.getLegalActions(agentIndex):
                #Calculate min value
                v = min(v, value(state.generateSuccessor(agentIndex, action), depth, nextAgent))
            #Return the min value
            return v

        #Initial values best action and best value
        bestAction = None
        bestValue = float('-inf')
        #Loop through legal actions of Pacman
        for action in gameState.getLegalActions(0):
            #Calculate the value of the action
            actionValue = value(gameState.generateSuccessor(0, action), 0, 1)
            #If the value is better than the best value
            if actionValue > bestValue:
                #Update the best value and best action
                bestValue = actionValue
                #Update the best action
                bestAction = action

        #Return the best action
        return bestAction

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Return the value of the state
        def value(state, depth, agentIndex, alpha, beta):
            #If state is terminal or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                #Return the evaluation function of the state
                return self.evaluationFunction(state)
            #If Pacman's turn (maximizing agent)
            if agentIndex == 0:
                #Return the max value of the state
                return maxValue(state, depth, alpha, beta)
            #If Ghosts' turn (minimizing agents)
            else:
                #Return the min value of the state
                return minValue(state, depth, agentIndex, alpha, beta)

        #Return the max value of the state
        def maxValue(state, depth, alpha, beta):
            #Initial value
            v = float('-inf')
            #Loop through legal actions of Pacman
            for action in state.getLegalActions(0):
                #Calculate max value
                v = max(v, value(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                #If value is greater than beta
                if v > beta:
                    #Return the value
                    return v
                #Update alpha
                alpha = max(alpha, v)
            #Return the max value
            return v

        #Return the min value of the state
        def minValue(state, depth, agentIndex, alpha, beta):
            #Initial value
            v = float('inf')
            #Get next agent
            nextAgent = agentIndex + 1
            #If next agent is the last agent
            if nextAgent == state.getNumAgents():
                #Reset next agent to 0
                nextAgent = 0
                #Increase depth
                depth += 1
            #Loop through legal actions of the agent
            for action in state.getLegalActions(agentIndex):
                #Calculate min value
                v = min(v, value(state.generateSuccessor(agentIndex, action), depth, nextAgent, alpha, beta))
                #If value is less than alpha
                if v < alpha:
                    #Return the value
                    return v
                #Update beta
                beta = min(beta, v)
            #Return the min value
            return v

        #Initial values for alpha, beta, best action and best value
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestValue = float('-inf')
        #Loop through legal actions of Pacman
        for action in gameState.getLegalActions(0):
            #Calculate the value of the action
            actionValue = value(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            #If the value is better than the best value
            if actionValue > bestValue:
                #Update the best value and best action
                bestValue = actionValue
                #Update the best action
                bestAction = action
            #Update alpha
            alpha = max(alpha, bestValue)

        #Return the best action
        return bestAction

        #util.raiseNotDefined()

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
        #Return the value of the state
        def value(state, depth, agentIndex):
            #If state is terminal or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                #Return the evaluation function of the state
                return self.evaluationFunction(state)
            #If Pacman's turn (maximizing agent)
            if agentIndex == 0:
                #Return the max value of the state
                return maxValue(state, depth)
            #If Ghosts' turn (minimizing agents)
            else:
                #Return the expected value of the state
                return expectValue(state, depth, agentIndex)

        #Return the max value of the state
        def maxValue(state, depth):
            #Initial value
            v = float('-inf')
            #Loop through legal actions of Pacman
            for action in state.getLegalActions(0):
                #Calculate max value
                v = max(v, value(state.generateSuccessor(0, action), depth, 1))
            #Return the max value
            return v
        
        #Return the expected value of the state
        def expectValue(state, depth, agentIndex):
            #Initial value
            v = 0
            #Get next agent
            nextAgent = agentIndex + 1
            #If next agent is the last agent
            if nextAgent == state.getNumAgents():
                #Reset next agent to 0
                nextAgent = 0
                #Increase depth
                depth += 1
            #Loop through legal actions of the agent
            for action in state.getLegalActions(agentIndex):
                #Calculate expected value (average of all values)
                v += value(state.generateSuccessor(agentIndex, action), depth, nextAgent)
            #Return the expected value
            return v / len(state.getLegalActions(agentIndex))

        #Initial values best action and best value
        bestAction = None
        bestValue = float('-inf')
        #Loop through legal actions of Pacman
        for action in gameState.getLegalActions(0):
            #Calculate the value of the action
            actionValue = value(gameState.generateSuccessor(0, action), 0, 1)
            #If the value is better than the best value
            if actionValue > bestValue:
                #Update the best value and best action
                bestValue = actionValue
                #Update the best action
                bestAction = action

        #Return the best action
        return bestAction

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This function evaluates the current game state for Pac-Man by assigning a score based on key factors. 
    It penalizes proximity to non-scared ghosts, rewards closeness to scared ghosts and food pellets, 
    and discourages leaving food uneaten, guiding Pac-Man to prioritize survival and food collection.
    One downside to this implementation is the fact that if a lone food pellet is further away, pacman has little incentive to go towards it.
    Thus, pacman will wait for a ghost to get near him, which (if lucky) will indirectly make him get closer to food. Then, the food incentive takes over again.
    """
    # Information extracted from the current GameState
    newPosition = currentGameState.getPacmanPosition()
    foodPellet = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    # Initial score from the successor game state
    evaluation = currentGameState.getScore()

    # Loop through ghost states
    for ghostState in ghostStates:
        # Get ghost current position
        ghostPosition = ghostState.getPosition()
        # Calculate Manhattan distance to ghost
        ghostDistance = util.manhattanDistance(newPosition, ghostPosition)

        # Decrease total score for closer non-scared ghosts
        if ghostDistance < 2:
            # Large penalty for being really close to a non-scared ghost
            evaluation -= 1/max(ghostDistance, sys.float_info.epsilon) # handle division by zero
        else:
            # If ghost is scared
            if ghostState.scaredTimer > 0:
                # Increase total score for closer scared ghosts
                evaluation += 10.0 / max(ghostDistance, sys.float_info.epsilon) # Avoid divide by zero error

    # Get list of food left in field
    remainingFood = foodPellet.asList()
    # If there is food left in field
    if remainingFood:
        # Calculate Manhattan distance to the closest food
        closestFoodDist = min([util.manhattanDistance(newPosition, food) for food in remainingFood])
        # Increase score for closer food
        evaluation += 1.0 / closestFoodDist * 0.1  # weight food more highly than running away
        # trying to encourage pacman to chase food that is further away
        #Calculate the Manhattan distance to the farthest food
        farthestFoodDist = max([util.manhattanDistance(newPosition, food) for food in remainingFood])
        #Increase score for being close to the farthest food
        if farthestFoodDist < 5:
            evaluation += 1.0 / max(farthestFoodDist, sys.float_info.epsilon)

    # Lower total score for remaining food dots
    evaluation -= len(remainingFood)
    
    # Return the score
    return evaluation

    # util.raiseNotDefined()
# Abbreviation
better = betterEvaluationFunction
