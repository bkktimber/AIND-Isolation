# Game Playing Agent Using Adversarial Search

In this project, I develop adversarial search agent to play the Isolation game. Beside of helper class provide by Udacity, for example, Isolation game rules, Isolation Board setup, I implement following techniques in `game_agent.py` to develop an agent:
   
 - `MinimaxPlayer.minimax()`: implement minimax search
 - `AlphaBetaPlayer.alphabeta()`: implement minimax search with alpha-beta pruning
 - `AlphaBetaPlayer.get_move()`: implement iterative deepening search
 - `custom_score()`, `custom_score_2()`, `custom_score_3()`: implement my own alternate position evaluation heuristic

Moreover, I also provide 2 analysis reports including:

 - Heuristic Analysis: detail analysis of evaluation heuristics I have implemented in `cutom_score()`, `cutom_score2()`, and `cutom_score3()`.
 - Research Review: review of Deep Blue, a Chess playing agent which beated World Chess Champion in 1997 developed by IBM.

 # Repository Structure

 - game_agent.py
 - Heuristic Review.pdf
 - research_review.pdf