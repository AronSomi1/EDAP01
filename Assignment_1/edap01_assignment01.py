import time
import copy
import math
import threading

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

    print("  " + " ".join(map(str, range(8)))) # Print column numbers
    for i, row in enumerate(board): # Print row numbers and the row itself
        print(f"{i} " + " ".join(row))

def is_valid_move(board, row, col, player):

    """
    Checks if a move is valid for the current player. Has the current game board, 
    coordinates of the move and the current player's color as input

    """

    if board[row][col] != ".": # Check if the cell is already occupied
        return False
    opponent = "B" if player == "W" else "W" # Determine the opponent's color
    # All 8 possible directions to check for valid moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for dr, dc in directions:
        r, c = row + dr, col + dc
        found_opponent = False
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r][c] == opponent: # Opponent's disk in this direction
                found_opponent = True
            elif board[r][c] == player:
                if found_opponent:
                    return True # Valid move
                break
            else: # Empty cell or end of the board
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

    return [(r, c) for r in range(8) for c in range(8) if is_valid_move(board, r, c, player)]

def apply_move(board, row, col, player):

    """
    Applies a move to the game board and flips the appropriate disks.
    Has the board, coordinates of the move and the player's color as imput.
    """

    opponent = "B" if player == "W" else "W"
    # All 8 possible direction to flip disks
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    board[row][col] = player # Place the player's disk
    for dr, dc in directions:
        r, c = row + dr, col + dc
        disks_to_flip = []
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
            disks_to_flip.append((r, c)) # Collect opponent's disks to flip
            r += dr
            c += dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            # If a player's disk is encountered, flip the collected disks
            for fr, fc in disks_to_flip:
                board[fr][fc] = player

def evaluate_board(board, player):

    """
    Evaluate the board by counting the difference in the number of disks.
    Has the current game board and the player's color as input.
    Returns an integer representing the evaluation score for the player.
    """

    opponent = "B" if player == "W" else "W"
    player_score = sum(row.count(player) for row in board)
    opponent_score = sum(row.count(opponent) for row in board)
    return player_score - opponent_score

def minimax(board, depth, alpha, beta, maximizing_player, player, start_time, time_limit):

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
        return evaluate_board(board, player), None

    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move[0], move[1], player)
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False, player, start_time, time_limit)
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
            eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True, player, start_time, time_limit)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha or time.time() - start_time > time_limit:
                break
        return min_eval, best_move

def computer_move(board, player, time_limit):

    """
    Computer move using minimax algorithm with a time limit.
    Has current game board, the computer's color and maximum time limit as inputs.
    Returns a touple representing the best move.
    """

    start_time = time.time()
    _, best_move = minimax(board, depth=10, alpha=-math.inf, beta=math.inf, 
                           maximizing_player=True, player=player, 
                           start_time=start_time, time_limit=time_limit)
    return best_move

def play_game():

    """
    Main function to play the game. Handles game flow, user input, and turns.
    """
    
    board = initialize_board()
    print("Welcome to Othello!")
    # Let the user choose their color
    human_player = None
    while human_player not in {"B", "W"}:
        human_player = input("Choose your color (B for dark, W for light): ").upper()
    computer_player = "B" if human_player == "W" else "W"

    # Set a time limit for the computer's response
    time_limit = None
    while not time_limit:
        try:
            time_limit = float(input("Enter time limit for computer's move (in seconds): "))
        except ValueError:
            print("Please enter a valid number.")

    current_player = "B"  # Dark always starts
    
    while True:
        print_board(board)
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:
            print(f"No valid moves for {current_player}. Passing turn.")
            current_player = computer_player if current_player == human_player else human_player
            if not get_valid_moves(board, current_player):
                # Game over: calculate and display scores
                print("No moves for either player. Game over!")
                human_score = sum(row.count(human_player) for row in board)
                computer_score = sum(row.count(computer_player) for row in board)
                print(f"Final score - {human_player}: {human_score}, {computer_player}: {computer_score}")
                if human_score > computer_score:
                    print(f"{human_player} wins!")
                elif human_score < computer_score:
                    print(f"{computer_player} wins!")
                else:
                    print("It's a draw!")
                break
            continue

        print(f"{current_player}'s turn. Valid moves: {valid_moves}")
        if current_player == human_player:
            # Human's turn
            move = None
            while move not in valid_moves:
                try:
                    row, col = map(int, input(f"{current_player}, enter your move (row col): ").split())
                    move = (row, col)
                except ValueError:
                    pass
        else:  # Computer's turn
            print("Computer is thinking...")
            move = computer_move(board, current_player, time_limit)
            print(f"Computer chooses move: {move}")

        if move:
            apply_move(board, move[0], move[1], current_player)
            current_player = computer_player if current_player == human_player else human_player

if __name__ == "__main__":
    play_game()
