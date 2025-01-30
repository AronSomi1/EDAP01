import time
import copy
import math
import threading
import random


def initialize_board():
    """
    Initialize playing board with the starting position.
    Function returns the game board.
    """

    board = [["." for _ in range(8)] for _ in range(8)]
    board[3][3] = "W"
    board[4][4] = "W"
    board[3][4] = "B"
    board[4][3] = "B"
    return board


def print_board(board):
    """
    Function prints the current state of the game board and has the board as input.
    """

    print("  " + " ".join(map(str, range(8))))  # Print column numbers
    for i, row in enumerate(board):  # Print row numbers and the row itself
        print(f"{i} " + " ".join(row))


def is_valid_move(board, row, col, player):
    """
    Checks if a move is valid for the current player. Has the current game board,
    coordinates of the move and the current player's color as input

    """

    if board[row][col] != ".":  # Check if the cell is already occupied
        return False
    opponent = "B" if player == "W" else "W"  # Determine the opponent's color
    # All 8 possible directions to check for valid moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        found_opponent = False
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c] == opponent:  # Opponent's disk in this direction
                found_opponent = True
            elif board[r][c] == player:
                if found_opponent:
                    return True  # Valid move
                break
            else:  # Empty cell or end of the board
                break
            r += dr
            c += dc
    return False


def get_valid_moves(board, player):
    """
    Finds all valid moves for the current player
    Has the current game board and the player's color as input.
    Returns a list of tuples representing valid moves.
    """

    return [
        (r, c) for r in range(8) for c in range(8) if is_valid_move(board, r, c, player)
    ]


def apply_move(board, row, col, player):
    """
    Applies a move to the game board and flips the appropriate disks.
    Has the board, coordinates of the move and the player's color as imput.
    """

    opponent = "B" if player == "W" else "W"
    # All 8 possible direction to flip disks
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    board[row][col] = player  # Place the player's disk
    for dr, dc in directions:
        r, c = row + dr, col + dc
        disks_to_flip = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
            disks_to_flip.append((r, c))  # Collect opponent's disks to flip
            r += dr
            c += dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            # If a player's disk is encountered, flip the collected disks
            for fr, fc in disks_to_flip:
                board[fr][fc] = player


def evaluate_advanced(board, player):
    """
    Improved heuristic: piece difference + corner control + mobility.
    - Pieces: Basic score (counts pieces)
    - Corners: Weight given to corner ownership
    - Mobility: Number of available moves
    """
    opponent = "B" if player == "W" else "W"

    # Piece Count
    player_pieces = sum(row.count(player) for row in board)
    opponent_pieces = sum(row.count(opponent) for row in board)
    piece_score = player_pieces - opponent_pieces

    # Corner Control (Corners are very strong positions)
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    player_corners = sum(1 for r, c in corners if board[r][c] == player)
    opponent_corners = sum(1 for r, c in corners if board[r][c] == opponent)
    corner_score = (player_corners - opponent_corners) * 5  # Weighted higher

    # Mobility (How many moves are available)
    player_moves = len(get_valid_moves(board, player))
    opponent_moves = len(get_valid_moves(board, opponent))
    mobility_score = (player_moves - opponent_moves) * 2  # Moderate weight

    # Final evaluation with weights
    return piece_score + corner_score + mobility_score


def evaluate_piece_count(board, player):
    """
    Evaluate the board by counting the difference in the number of disks.
    Has the current game board and the player's color as input.
    Returns an integer representing the evaluation score for the player.
    """

    opponent = "B" if player == "W" else "W"
    player_score = sum(row.count(player) for row in board)
    opponent_score = sum(row.count(opponent) for row in board)
    return player_score - opponent_score


def minimax(
    board,
    depth,
    alpha,
    beta,
    maximizing_player,
    player,
    start_time,
    time_limit,
    eval_fun,
):
    """
    Implements the minimax algorithm with alpha-beta pruning and time-limiting.
    Has the current game board, max. depht of the search, alpha/beta pruning bounds,
    maximizing_player (true if maximizing player's turn, else false), current player's color,
    starting time when the move calculation started and time limit as inputs.
    Returns a touple of evaluation score and best move.
    """

    opponent = "B" if player == "W" else "W"
    valid_moves = get_valid_moves(board, player if maximizing_player else opponent)

    # Base case: return evaluation score at max depth, no moves, or timeout
    if depth == 0 or not valid_moves or time.time() - start_time > time_limit:
        return eval_fun(board, player), None

    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], player)
            eval_score, _ = minimax(
                new_board,
                depth - 1,
                alpha,
                beta,
                False,
                player,
                start_time,
                time_limit,
                eval_fun,
            )
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha or time.time() - start_time > time_limit:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in valid_moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], opponent)
            eval_score, _ = minimax(
                new_board,
                depth - 1,
                alpha,
                beta,
                True,
                player,
                start_time,
                time_limit,
                eval_fun,
            )
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha or time.time() - start_time > time_limit:
                break
        return min_eval, best_move


def random_move(board, player):
    """
    Selects a random valid move for the player.
    """
    valid_moves = get_valid_moves(board, player)
    return random.choice(valid_moves) if valid_moves else None


def computer_move(board, player, time_limit, eval_fun):
    """
    Computer move using minimax algorithm with a time limit.
    Has current game board, the computer's color and maximum time limit as inputs.
    Returns a touple representing the best move.
    """

    start_time = time.time()
    _, best_move = minimax(
        board,
        depth=10,
        alpha=-math.inf,
        beta=math.inf,
        maximizing_player=True,
        player=player,
        start_time=start_time,
        time_limit=time_limit,
        eval_fun=eval_fun,
    )
    return best_move


def choose_player_type(player_name):
    """
    Asks the user to choose the type of player: human, minimax, or random.
    """
    while True:
        player_type = (
            input(f"Choose {player_name} type (human/minimax/random): ").strip().lower()
        )
        if player_type in {"human", "minimax", "random"}:
            return player_type
        print("Invalid input. Please enter 'human', 'minimax', or 'random'.")


def choose_time_limit():
    """
    Asks the user to input a time limit for the Minimax AI.
    """
    while True:
        try:
            time_limit = float(input("Enter time limit for Minimax AI (in seconds): "))
            return time_limit
        except ValueError:
            print("Invalid input. Please enter a number.")


def choose_evaluation_function():
    """
    Asks the user to choose an evaluation function for the Minimax AI.
    """
    eval_functions = {"1": evaluate_piece_count, "2": evaluate_advanced}

    print("Choose evaluation function for Minimax:")
    print("1: Simple Piece Count")
    print("2: Advanced Heuristic (Piece Count + Corner Control + Mobility)")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in eval_functions:
            return eval_functions[choice]
        print("Invalid input. Please enter '1' or '2'.")


def play_game():
    """
    Interactive version of Othello that allows players to choose AI types, time limits,
    and evaluation functions for Minimax AI.
    """
    board = initialize_board()
    print("Welcome to Othello!")

    # Let the user select player types
    player1_type = choose_player_type("Player 1 (Black)")
    player2_type = choose_player_type("Player 2 (White)")

    # Set evaluation functions and time limits for Minimax
    eval_fn_p1 = None
    eval_fn_p2 = None
    time_limit_p1 = None
    time_limit_p2 = None

    if player1_type == "minimax":
        print("Selecting for player 1, Minimax AI.")
        time_limit_p1 = choose_time_limit()
        eval_fn_p1 = choose_evaluation_function()

    if player2_type == "minimax":
        print("Selecting for player 2, Minimax AI.")
        time_limit_p2 = choose_time_limit()
        eval_fn_p2 = choose_evaluation_function()

    current_player = "B"  # Black always starts
    player1 = "B"
    player2 = "W"

    while True:
        print_board(board)
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:
            print(f"No valid moves for {current_player}. Passing turn.")
            current_player = player2 if current_player == player1 else player1
            if not get_valid_moves(board, current_player):
                print("No moves for either player. Game over!")
                player1_score = sum(row.count(player1) for row in board)
                player2_score = sum(row.count(player2) for row in board)
                print(
                    f"Final score - {player1}: {player1_score}, {player2}: {player2_score}"
                )
                if player1_score > player2_score:
                    print(f"{player1} wins!")
                elif player1_score < player2_score:
                    print(f"{player2} wins!")
                else:
                    print("It's a draw!")
                break
            continue

        print(f"{current_player}'s turn. Valid moves: {valid_moves}")

        # Determine player type
        if current_player == player1:
            player_type = player1_type
            eval_fn = eval_fn_p1
            time_limit = time_limit_p1
        else:
            player_type = player2_type
            eval_fn = eval_fn_p2
            time_limit = time_limit_p2

        # Player makes a move based on their type
        if player_type == "human":
            move = None
            while move not in valid_moves:
                try:
                    row, col = map(
                        int,
                        input(f"{current_player}, enter your move (row col): ").split(),
                    )
                    move = (row, col)
                except ValueError:
                    print(
                        "Invalid move. Enter row and column as two numbers separated by space."
                    )
        elif player_type == "minimax":
            print("Minimax AI is thinking...")
            move = computer_move(board, current_player, time_limit, eval_fn)
            print(f"Minimax AI chooses move: {move}")
            print("Evaluation score:", eval_fn(board, current_player))
        elif player_type == "random":
            print("Random AI is making a move...")
            move = random_move(board, current_player)
            print(f"Random AI chooses move: {move}")

        if move:
            apply_move(board, move[0], move[1], current_player)
            current_player = player2 if current_player == player1 else player1


if __name__ == "__main__":
    play_game()
